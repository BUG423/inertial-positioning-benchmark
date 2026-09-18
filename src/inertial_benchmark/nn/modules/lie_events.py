"""SE(3) 李群事件与事件堆叠（`models/nio_lie_events` 使用）。

规格见 `docs/algorithms/nio_lie_events.md` §2.1–2.2。本模块只做**输入表示**：把一个窗口的
世界系 IMU、逐样本姿态与起点速度变成 12 通道的事件堆叠 ``(B, 12, bins)``，通道顺序为
``[ω_x, ω_y, ω_z, a_x, a_y, a_z, ρ̂_x, ρ̂_y, ρ̂_z, φ̂_x, φ̂_y, φ̂_z]``（世界系；论文写作加计在前，
**以官方代码为准**取陀螺在前）。

约定（卡 §2.1，Sophus 顺序 ``ξ = [ρ(3), φ(3)]``，平移与旋转之间**不加权**）：

* ``Exp(ξ) = (exp(φ), J_l(φ)·ρ)``，``Log(T) = (J_l(φ)⁻¹·t, φ)``（本模块按 ``[ρ, φ]`` 排列）；
* ``exp(φ)`` 用 Rodrigues 公式，``‖φ‖ < 1e-10`` 时取 ``I + [φ]×``；
* ``log(R)``：``x = clip((tr R − 1)/2, −1, 1)``、``θ = arccos x``；**``|θ − π| < 1e-3`` 时直接返回
  零向量**（官方行为，不连续，由单元测试覆盖）；否则 ``φ = ½·vee(R − Rᵀ)·θ/sin θ``，
  ``|θ| < 1e-3`` 时 ``sin θ/θ`` 取 1；
* ``J_l``、``J_l⁻¹`` 的小角度分支阈值为 1e-5。

与官方实现的差异（卡 §6、§10-5）：``‖w‖ = 0``（完全静止或零位移）时官方 ``u = w/0`` 产生 NaN
并污染此后所有事件；本实现在该情况下取 ``u = 0``、``n = 0``（不产生事件、参考位姿不动），
保证输出有限，并由单元测试锁定。
"""

from __future__ import annotations

from typing import Optional

import torch

GRAVITY_VECTOR = (0.0, 0.0, -9.81)
EPS_EXP = 1e-10          # exp(φ) 的小角度阈值（官方值）
EPS_LOG_PI = 1e-3        # |θ − π| < EPS_LOG_PI 时 log(R) 返回 0（官方行为）
EPS_LOG_SMALL = 1e-3     # |θ| < EPS_LOG_SMALL 时 sin(θ)/θ 取 1（官方行为）
EPS_JACOBIAN = 1e-5      # J_l / J_l⁻¹ 的小角度阈值（官方值）
EPS_NORM = 1e-12         # 归一化的零范数保护（IPB 修正，官方会产生 NaN）
POLARITY_EPS = 1e-4      # 每个桶内极性归一化的分母偏移（官方值）


def skew(vec: torch.Tensor) -> torch.Tensor:
    """``(..., 3) → (..., 3, 3)`` 的反对称矩阵 ``[v]×``。"""
    zero = torch.zeros_like(vec[..., 0])
    x, y, z = vec[..., 0], vec[..., 1], vec[..., 2]
    rows = [torch.stack([zero, -z, y], dim=-1),
            torch.stack([z, zero, -x], dim=-1),
            torch.stack([-y, x, zero], dim=-1)]
    return torch.stack(rows, dim=-2)


def quat_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    """单位四元数 ``(..., 4)``（``wxyz``，body → world）→ 旋转矩阵 ``(..., 3, 3)``（torch 版）。"""
    quat = quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp_min(EPS_NORM)
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    rows = [
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ]
    return torch.stack(rows, dim=-2)


def so3_exp(phi: torch.Tensor) -> torch.Tensor:
    """Rodrigues 公式；``‖φ‖ < 1e-10`` 时取一阶近似 ``I + [φ]×``。"""
    theta = torch.linalg.vector_norm(phi, dim=-1, keepdim=True).unsqueeze(-1)
    cross = skew(phi)
    eye = torch.eye(3, dtype=phi.dtype, device=phi.device).expand(cross.shape)
    safe = theta.clamp_min(EPS_EXP)
    full = eye + torch.sin(safe) / safe * cross \
        + (1.0 - torch.cos(safe)) / safe**2 * (cross @ cross)
    return torch.where(theta < EPS_EXP, eye + cross, full)


def so3_log(rot: torch.Tensor) -> torch.Tensor:
    """``log(R)``，含官方的两处特判（``|θ−π| < 1e-3`` 返回 0，``|θ| < 1e-3`` 时 ``sinc`` 取 1）。"""
    trace = rot[..., 0, 0] + rot[..., 1, 1] + rot[..., 2, 2]
    cosine = torch.clamp(0.5 * (trace - 1.0), -1.0, 1.0)
    theta = torch.arccos(cosine)
    antisymmetric = rot - rot.transpose(-1, -2)
    vee = 0.5 * torch.stack([antisymmetric[..., 2, 1], antisymmetric[..., 0, 2],
                             antisymmetric[..., 1, 0]], dim=-1)
    small = theta.abs() < EPS_LOG_SMALL
    sinc = torch.where(small, torch.ones_like(theta), torch.sin(theta) / theta.clamp_min(1e-30))
    phi = vee / sinc.clamp_min(1e-30).unsqueeze(-1)
    near_pi = ((theta - torch.pi).abs() < EPS_LOG_PI).unsqueeze(-1)
    return torch.where(near_pi, torch.zeros_like(phi), phi)


def left_jacobian(phi: torch.Tensor) -> torch.Tensor:
    """SO(3) 左雅可比 ``J_l(φ)``；``θ < 1e-5`` 时取 ``I + ½Φ``。"""
    theta = torch.linalg.vector_norm(phi, dim=-1, keepdim=True).unsqueeze(-1)
    cross = skew(phi)
    eye = torch.eye(3, dtype=phi.dtype, device=phi.device).expand(cross.shape)
    safe = theta.clamp_min(EPS_JACOBIAN)
    full = eye + (1.0 - torch.cos(safe)) / safe**2 * cross \
        + (safe - torch.sin(safe)) / safe**3 * (cross @ cross)
    return torch.where(theta < EPS_JACOBIAN, eye + 0.5 * cross, full)


def left_jacobian_inverse(phi: torch.Tensor) -> torch.Tensor:
    """``J_l⁻¹(φ)``；``θ < 1e-5`` 时取 ``I − ½Φ + Φ²/12``。"""
    theta = torch.linalg.vector_norm(phi, dim=-1, keepdim=True).unsqueeze(-1)
    cross = skew(phi)
    eye = torch.eye(3, dtype=phi.dtype, device=phi.device).expand(cross.shape)
    safe = theta.clamp_min(EPS_JACOBIAN)
    half = 0.5 * safe
    coefficient = (1.0 - half * torch.cos(half) / torch.sin(half).clamp_min(1e-30)) / safe**2
    full = eye - 0.5 * cross + coefficient * (cross @ cross)
    return torch.where(theta < EPS_JACOBIAN,
                       eye - 0.5 * cross + (cross @ cross) / 12.0, full)


def se3_exp(xi: torch.Tensor) -> tuple:
    """``ξ = [ρ, φ]``（``(..., 6)``）→ ``(R, t)``，``t = J_l(φ)·ρ``。"""
    rho, phi = xi[..., :3], xi[..., 3:]
    rot = so3_exp(phi)
    trans = (left_jacobian(phi) @ rho.unsqueeze(-1)).squeeze(-1)
    return rot, trans


def se3_log(rot: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
    """``(R, t) → ξ = [J_l(φ)⁻¹·t, φ]``（``(..., 6)``）。"""
    phi = so3_log(rot)
    rho = (left_jacobian_inverse(phi) @ trans.unsqueeze(-1)).squeeze(-1)
    return torch.cat([rho, phi], dim=-1)


def se3_compose(rot_a: torch.Tensor, trans_a: torch.Tensor, rot_b: torch.Tensor,
                trans_b: torch.Tensor) -> tuple:
    """``T_a · T_b``。"""
    return rot_a @ rot_b, (rot_a @ trans_b.unsqueeze(-1)).squeeze(-1) + trans_a


def se3_inverse_compose(rot_a: torch.Tensor, trans_a: torch.Tensor, rot_b: torch.Tensor,
                        trans_b: torch.Tensor) -> tuple:
    """``T_a⁻¹ · T_b``。"""
    rot = rot_a.transpose(-1, -2)
    return rot @ rot_b, (rot @ (trans_b - trans_a).unsqueeze(-1)).squeeze(-1)


def _normalise(vec: torch.Tensor) -> tuple:
    """返回 ``(单位向量, 范数)``；范数为 0 时单位向量取 0（官方在这里产生 NaN，卡 §10-5）。"""
    norm = torch.linalg.vector_norm(vec, dim=-1, keepdim=True)
    unit = torch.where(norm > EPS_NORM, vec / norm.clamp_min(EPS_NORM),
                       torch.zeros_like(vec))
    return unit, norm.squeeze(-1)


def generate_event_stack(times: torch.Tensor, measurements: torch.Tensor,
                         polarities: torch.Tensor, bins: int) -> torch.Tensor:
    """按**序号**分桶（官方 ``generate_event_stack``，与式 26–27 一致，不看时间戳）。

    第 ``j`` 个事件（从 0 起，按 ``times`` 排序后）落入桶 ``⌊j·(B−1)/(M−1)⌋``；
    ``E_{ω,a}[b]`` 为桶内量测均值（空桶为 0），``E_p[b] = Ē_p/(‖Ē_p‖ + 1e-4)``
    （6 维**联合**归一化，空桶为 0）。返回 ``(12, bins)``。
    """
    count = times.shape[0]
    stack = torch.zeros(12, bins, dtype=measurements.dtype, device=measurements.device)
    if count == 0:
        return stack
    order = torch.argsort(times)
    measurements, polarities = measurements[order], polarities[order]
    if count == 1:
        index = torch.zeros(1, dtype=torch.long, device=times.device)
    else:
        grid = torch.linspace(0.0, float(bins - 1), count, dtype=torch.float64,
                              device=times.device)
        index = grid.to(torch.int64)
    totals = torch.zeros(bins, dtype=measurements.dtype, device=measurements.device)
    totals.index_add_(0, index, torch.ones_like(times))
    summed = torch.zeros(bins, 6, dtype=measurements.dtype, device=measurements.device)
    summed.index_add_(0, index, measurements)
    polar = torch.zeros(bins, 6, dtype=polarities.dtype, device=polarities.device)
    polar.index_add_(0, index, polarities)
    nonempty = totals > 0
    divisor = totals.clamp_min(1.0).unsqueeze(-1)
    mean_measurement = summed / divisor
    mean_polarity = polar / divisor
    norm = torch.linalg.vector_norm(mean_polarity, dim=-1, keepdim=True)
    unit_polarity = mean_polarity / (norm + POLARITY_EPS)
    keep = nonempty.unsqueeze(-1)
    stack[:6] = torch.where(keep, mean_measurement, torch.zeros_like(mean_measurement)).T
    stack[6:] = torch.where(keep, unit_polarity, torch.zeros_like(unit_polarity)).T
    return stack


def lie_events(gyro_world: torch.Tensor, acc_world: torch.Tensor, rotation: torch.Tensor,
               init_velocity: torch.Tensor, dt: float, threshold: float, bins: int = 200,
               max_events_per_step: int = 64,
               gravity: Optional[tuple] = None) -> dict:
    """一个窗口批的 SE(3) 李事件与事件堆叠（卡 §2.2 的 8 步）。

    参数（全部在**同一个重力对齐系**中，IPB 的视图坐标系即可）：

    * ``gyro_world``/``acc_world``：``(B, T, 3)``，世界系角速度与比力；
    * ``rotation``：``(B, T, 3, 3)``，逐样本姿态 ``R_i``（body → 该世界系）；
    * ``init_velocity``：``(B, 3)``，窗口起点速度 ``v0``（**特权输入**，来自参考真值）；
    * ``dt``：采样间隔（IPB 的 200 Hz 网格严格均匀）；``threshold``：阈值 θ。

    返回 ``{"stack": (B, 12, bins), "counts": (B,) 真实事件数, "pose_rotation"/"pose_translation":
    末端预积分位姿}``。计算全程 ``no_grad``（事件生成是输入表示，不需要对 IMU 求导）。
    """
    device, dtype = gyro_world.device, gyro_world.dtype
    batch, steps = gyro_world.shape[0], gyro_world.shape[1]
    g = torch.tensor(gravity or GRAVITY_VECTOR, dtype=dtype, device=device)
    # 1. 机体系量测
    gyro_body = torch.einsum("btji,btj->bti", rotation, gyro_world)
    acc_body = torch.einsum("btji,btj->bti", rotation, acc_world)
    # 2/3. 预积分 + 测地线水平穿越
    rot_k = rotation[:, 0].clone()
    pos_k = torch.zeros(batch, 3, dtype=dtype, device=device)
    vel_k = init_velocity.to(dtype)
    rot_t, pos_t = rot_k.clone(), pos_k.clone()
    rot_ref, pos_ref = rot_k.clone(), pos_k.clone()
    counts = torch.zeros(batch, dtype=torch.int64, device=device)
    rows: list = []      # [(mask (B,), times (B,), meas (B,6), polarity (B,6))]
    for i in range(1, steps):
        rot_next = rot_k @ so3_exp(gyro_body[:, i] * dt)
        acc_w = (rot_k @ acc_body[:, i].unsqueeze(-1)).squeeze(-1)
        vel_next = vel_k + (acc_w + g) * dt
        pos_next = pos_k + vel_k * dt + 0.5 * (acc_w + g) * dt**2
        rel_rot, rel_trans = se3_inverse_compose(rot_t, pos_t, rot_next, pos_next)
        direction, span = _normalise(se3_log(rel_rot, rel_trans))
        ref_rot, ref_trans = se3_inverse_compose(rot_ref, pos_ref, rot_next, pos_next)
        polarity_unit, distance = _normalise(se3_log(ref_rot, ref_trans))
        crossings = torch.floor(distance / threshold).to(torch.int64)
        crossings = crossings.clamp(0, int(max_events_per_step))
        counts = counts + crossings
        limit = int(crossings.max()) if batch else 0
        cursor_rot = rot_ref
        for j in range(1, limit + 1):
            mask = crossings >= j
            step_rot, step_trans = se3_exp(polarity_unit * (j * threshold))
            event_rot, event_trans = se3_compose(rot_ref, pos_ref, step_rot, step_trans)
            offset = se3_log(*se3_inverse_compose(rot_t, pos_t, event_rot, event_trans))
            beta = (offset * direction).sum(-1) / span.clamp_min(EPS_NORM)
            interpolated = torch.cat([
                gyro_world[:, i - 1] + (gyro_world[:, i] - gyro_world[:, i - 1])
                * beta.unsqueeze(-1),
                acc_world[:, i - 1] + (acc_world[:, i] - acc_world[:, i - 1])
                * beta.unsqueeze(-1)], dim=-1)
            # 极性：同一个 u 分两段，各用**上一个**参考位姿的旋转转到世界系
            polarity = torch.cat([
                (cursor_rot @ polarity_unit[:, :3].unsqueeze(-1)).squeeze(-1),
                (cursor_rot @ polarity_unit[:, 3:].unsqueeze(-1)).squeeze(-1)], dim=-1)
            rows.append((mask, (i - 1 + beta) * dt, interpolated, polarity))
            cursor_rot = event_rot
        total_rot, total_trans = se3_exp(polarity_unit * (crossings.to(dtype) * threshold)
                                         .unsqueeze(-1))
        rot_ref, pos_ref = se3_compose(rot_ref, pos_ref, total_rot, total_trans)
        rot_k, pos_k, vel_k = rot_next, pos_next, vel_next
        rot_t, pos_t = rot_next, pos_next
    # 4–7. 首尾伪事件 + 排序 + 按序号分桶
    final_direction, _ = _normalise(se3_log(*se3_inverse_compose(rotation[:, 0],
                                                                 torch.zeros_like(pos_k),
                                                                 rot_k, pos_k)))
    fallback = torch.cat([
        (rotation[:, 0] @ final_direction[:, :3].unsqueeze(-1)).squeeze(-1),
        (rotation[:, 0] @ final_direction[:, 3:].unsqueeze(-1)).squeeze(-1)], dim=-1)
    first_measurement = torch.cat([gyro_world[:, 0], acc_world[:, 0]], dim=-1)
    last_measurement = torch.cat([gyro_world[:, -1], acc_world[:, -1]], dim=-1)
    last_time = (steps - 1) * dt
    stacks = []
    for b in range(batch):
        selected = [(float(t[b]), m[b], p[b]) for mask, t, m, p in rows if bool(mask[b])]
        selected.sort(key=lambda item: item[0])
        first_polarity = selected[0][2] if selected else fallback[b]
        last_polarity = selected[-1][2] if selected else fallback[b]
        times = [0.0] + [item[0] for item in selected] + [last_time]
        measurements = [first_measurement[b]] + [item[1] for item in selected] \
            + [last_measurement[b]]
        polarities = [first_polarity] + [item[2] for item in selected] + [last_polarity]
        stacks.append(generate_event_stack(
            torch.tensor(times, dtype=dtype, device=device),
            torch.stack(measurements), torch.stack(polarities), bins))
    return {"stack": torch.stack(stacks), "counts": counts,
            "pose_rotation": rot_k, "pose_translation": pos_k}
