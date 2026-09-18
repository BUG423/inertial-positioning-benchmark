"""数据划分（DESIGN 2.4）：官方划分优先、按 ``group_id`` 分组分层抽取 val、泄漏检查。

划分文件分两个“族”：默认族（``train/val/test`` 与 ``test_*`` 子集，训练与评测使用）和
官方泄漏族（``official_*``，仅当转换器声明官方划分存在泄漏时写出，只用于与文献对照）。
泄漏检查在各族内部分别进行，两族之间的重叠是预期的，不检查。

生成 val 时除“同一 ``group_id`` 不跨划分”之外还有第二条底线（DESIGN 2.4）：
**val 出现的条件必须在 train 里也出现**。只属于一个 group 的条件（例如 RIDI 只有受试者
``ma`` 有 ``bag_low`` 携带方式）一旦被抽进 val，这个条件就从 train 里彻底消失，模型无从学起，
val 上的误差不再反映“选得好不好”而是“外推得多差”。``stratified_group_holdout`` 因此
在分组抽样前先按条件（``conditions``）过滤候选组，并按条件分层保证 val 覆盖尽量多的条件。
"""

from __future__ import annotations

import re
from itertools import combinations
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Union

import numpy as np

PathLike = Union[str, Path]
MAIN_SPLITS = ("train", "val", "test")
# 名字含这些词的官方子集表示“未见过的受试者/场景”，与训练集共享 group 视为泄漏
UNSEEN_TOKENS = ("unseen", "unknown", "novel")
# 转换器声明官方划分泄漏时，官方划分以该前缀另存（DESIGN 2.4）
OFFICIAL_PREFIX = "official_"
# 划分审计与分层抽样的默认条件维度（数据卡可用 ``strata:`` 覆盖，见 DESIGN 2.4）
DEFAULT_STRATA: tuple = ("placement", "device_id", "position_source")
# val 相对目标比例允许的超额（0.5 → 目标 10% 时最多 15%）
DEFAULT_SIZE_TOLERANCE = 0.5
# 由划分审计自动派生的分布外测试子集前缀（原样单列，绝不混进主 test）
OOD_PREFIX = "test_ood_"


def split_family(name: str) -> str:
    """划分所属的族：``official_`` 前缀为官方泄漏族，其余为默认族（空串）。"""
    return OFFICIAL_PREFIX if name.startswith(OFFICIAL_PREFIX) else ""


def read_split(path: PathLike) -> list:
    """读取划分文件：每行一个 ``sequence_id``，忽略空行与 ``#`` 注释。"""
    ids = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            ids.append(line)
    return ids


def write_split(path: PathLike, ids: Iterable[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{i}\n" for i in ids), encoding="utf-8")


def load_splits(root: PathLike) -> dict:
    """读取 ``<root>/splits/*.txt``，返回 ``{split_name: [ids]}``。"""
    folder = Path(root) / "splits"
    if not folder.is_dir():
        return {}
    return {p.stem: read_split(p) for p in sorted(folder.glob("*.txt"))}


def normalise_strata(spec: Optional[Iterable] = None) -> list:
    """把数据卡声明的分层维度规范化为 ``[{'name', 'kind', ...}]``（DESIGN 2.4）。

    三种维度：

    * ``"placement"`` 或 ``{"attr": "placement"}``：**属性维度**，直接取序列元信息的该字段；
    * ``{"name": "motion_variant", "of": "sequence_id", "pattern": r"..."}``：**正则维度**，
      对 ``of`` 字段（默认 ``sequence_id``）取第一个捕获组——条件只写在名字里时用
      （RIDI 的 ``bag_low``/``leg_front`` 只出现在序列名中）；
    * ``{"name": "speed_bin", "of": "mean_speed", "quantiles": 3}``：**分位维度**，把数值字段
      分成等频区间。分位维度只用于“让 val 覆盖该分布”，不参与“val 条件必须在 train 出现”
      的硬判定（连续量的分箱边界是人为的，不是可学/不可学的条件）。

    ``None`` 取 ``DEFAULT_STRATA``。
    """
    out = []
    for item in (DEFAULT_STRATA if spec is None else spec):
        if isinstance(item, str):
            out.append({"name": item, "kind": "attr", "of": item})
            continue
        if not isinstance(item, Mapping):
            raise ValueError(f"stratum spec must be a string or a mapping, got {item!r}")
        item = dict(item)
        if item.get("kind") in ("attr", "regex", "quantile"):
            out.append(item)  # 已规范化（``normalise_strata`` 可重复调用）
        elif "attr" in item:
            out.append({"name": str(item.get("name", item["attr"])), "kind": "attr",
                        "of": str(item["attr"])})
        elif "pattern" in item:
            out.append({"name": str(item["name"]), "kind": "regex",
                        "of": str(item.get("of", "sequence_id")), "pattern": str(item["pattern"])})
        elif "quantiles" in item:
            out.append({"name": str(item["name"]), "kind": "quantile",
                        "of": str(item.get("of", "mean_speed")),
                        "quantiles": int(item["quantiles"])})
        else:
            raise ValueError(f"stratum {item!r} needs one of 'attr', 'pattern', 'quantiles'")
    names = [d["name"] for d in out]
    duplicate = sorted({n for n in names if names.count(n) > 1})
    if duplicate:
        raise ValueError(f"duplicate stratum names: {duplicate}")
    return out


def _field(sequence_id: str, meta: Mapping, field: str):
    if field == "sequence_id":
        return sequence_id
    if field == "mean_speed":
        duration = float(meta.get("duration_s") or 0.0)
        return float(meta.get("distance_m") or 0.0) / duration if duration > 0 else float("nan")
    return meta.get(field)


def build_conditions(
    ids: Iterable[str],
    meta: Mapping[str, Mapping],
    strata: Optional[Iterable] = None,
    groups: Optional[Mapping[str, str]] = None,
) -> tuple:
    """为每条序列算出条件标签元组，返回 ``(conditions, info)``。

    标签形如 ``"placement=bag"``；缺失的属性记为 ``unknown``，正则不匹配时退化为原字段值。
    与 ``group_id`` 等价的维度（每个取值都只出现在一个组里，例如 TLIO 的 ``device_id``）被丢弃：
    分组划分下这种维度必然“val 有 train 没有”，留着只会掩盖真正的问题。
    ``info`` 记录规范化后的维度、丢弃原因、只用于覆盖的维度名与每个维度的取值直方图。
    """
    ids = list(ids)
    dims = normalise_strata(strata)
    values: dict = {d["name"]: {} for d in dims}
    for d in dims:
        name, kind = d["name"], d["kind"]
        if kind == "quantile":
            raw = {i: _field(i, meta.get(i, {}), d["of"]) for i in ids}
            finite = np.asarray([v for v in raw.values() if v is not None and np.isfinite(v)],
                                dtype=float)
            edges = (np.quantile(finite, np.linspace(0.0, 1.0, d["quantiles"] + 1)[1:-1])
                     if finite.size else np.empty(0))
            d["edges"] = [round(float(e), 6) for e in edges]
            for i in ids:
                v = raw[i]
                values[name][i] = ("unknown" if v is None or not np.isfinite(v)
                                   else f"q{int(np.searchsorted(edges, v, side='right'))}")
            continue
        for i in ids:
            raw = _field(i, meta.get(i, {}), d["of"])
            text = "unknown" if raw is None else str(raw)
            if kind == "regex":
                match = re.match(d["pattern"], text)
                text = match.group(1) if match and match.groups() else text
            values[name][i] = text or "unknown"
    dropped = {}
    if groups is not None:
        for d in list(dims):
            name = d["name"]
            per_value: dict = {}
            for i in ids:
                per_value.setdefault(values[name][i], set()).add(str(groups.get(i, i)))
            if per_value and all(len(g) == 1 for g in per_value.values()):
                dropped[name] = "every value belongs to exactly one group_id (no extra information)"
                dims.remove(d)
                values.pop(name)
    conditions = {i: tuple(f"{d['name']}={values[d['name']][i]}" for d in dims) for i in ids}
    info = {
        "strata": [dict(d) for d in dims],
        "coverage_only": [d["name"] for d in dims if d["kind"] == "quantile"],
        "dropped": dropped,
        "values": {d["name"]: sorted({values[d["name"]][i] for i in ids}) for d in dims},
    }
    return conditions, info


def _condition_dim(label: str) -> str:
    return label.split("=", 1)[0]


def stratified_group_holdout(
    ids: Iterable[str],
    groups: Mapping[str, str],
    conditions: Mapping[str, Sequence[str]],
    fraction: float = 0.1,
    seed: int = 0,
    weights: Optional[Mapping[str, float]] = None,
    *,
    tolerance: float = DEFAULT_SIZE_TOLERANCE,
    coverage_only: Iterable[str] = (),
    avoid_groups: Iterable[str] = (),
) -> tuple:
    """按条件分层的分组抽样：抽 ``fraction`` 的权重做 val，同时保证 val 的条件被 train 覆盖。

    确定性算法（``seed`` 只影响第 3 步的候选顺序）：

    1. **资格**：组 ``g`` 可进 val，当且仅当它的每个硬条件（不在 ``coverage_only`` 里的维度）
       在剩余的 train 组里至少还有一个组提供，且 train 至少留下一个组。每选一个组就重新计数，
       所以两个组共有的唯一条件不会被同时抽走。
    2. **覆盖**：条件按权重从大到小遍历；还没被 val 覆盖的条件，从合格候选里取
       ``(是否出现在 test, 组权重, 组名)`` 最小的那个——优先不与 test 共享组的组，
       其次取最小的组，好把预算留给别的条件。累计权重超过
       ``(1 + tolerance) × 目标`` 的候选跳过；预算不够的条件如实记为未覆盖。
    3. **补足**：若累计权重仍低于目标，按 ``seed`` 打乱后的顺序（同样先不与 test 共享组的组）
       逐个尝试，只有让累计权重更接近目标且不超上限时才加入。
    4. **兜底**：若一个组都没选上，取最接近目标的合格组（即使超上限），并在 ``info`` 里警告。

    返回 ``(train_ids, val_ids, info)``；``info`` 记录方法、种子、被排除的组及其独占条件、
    val 未覆盖的条件与实际比例，全部写进 ``dataset.json`` 的 ``split_method``。
    """
    ids = list(ids)
    info: dict = {"method": "stratified_group_holdout", "key": "group_id", "fraction": fraction,
                  "seed": seed, "weighting": "duration" if weights else "count",
                  "size_tolerance": tolerance}
    if not ids:
        return [], [], {**info, "warning": "no sequences"}
    w = {i: float(weights.get(i, 1.0)) if weights else 1.0 for i in ids}
    key = {i: str(groups.get(i, i)) for i in ids}
    unique = sorted(set(key.values()))
    coverage_only = set(coverage_only)
    if len(unique) < 2:
        train, val, legacy = group_holdout(ids, groups, fraction, seed, weights)
        return train, val, {**info, **legacy,
                            "warning": "only one group: fell back to a sequence-level split "
                                       "(leakage possible)"}

    group_weight: dict = {g: 0.0 for g in unique}
    group_conditions: dict = {g: set() for g in unique}
    condition_weight: dict = {}
    for i in ids:
        group_weight[key[i]] += w[i]
        labels = tuple(conditions.get(i, ()))
        group_conditions[key[i]].update(labels)
        for label in labels:
            condition_weight[label] = condition_weight.get(label, 0.0) + w[i]
    train_count = {label: 0 for label in condition_weight}
    for g in unique:
        for label in group_conditions[g]:
            train_count[label] += 1
    hard = {g: {c for c in group_conditions[g] if _condition_dim(c) not in coverage_only}
            for g in unique}
    owned = {g: sorted(c for c in hard[g] if train_count[c] < 2) for g in unique}
    info["ineligible_groups"] = {g: v for g, v in sorted(owned.items()) if v}

    total = sum(w.values())
    target = fraction * total
    cap = (1.0 + tolerance) * target
    avoid = set(avoid_groups)
    rank = {g: (1 if g in avoid else 0) for g in unique}
    chosen: list = []
    acc = 0.0

    def eligible(g: str) -> bool:
        if g in chosen or len(chosen) + 1 >= len(unique):
            return False
        return all(train_count[c] - 1 >= 1 for c in hard[g])

    def take(g: str) -> None:
        nonlocal acc
        chosen.append(g)
        acc += group_weight[g]
        for label in group_conditions[g]:
            train_count[label] -= 1

    covered: set = set()
    unmet: list = []
    for label in sorted(condition_weight, key=lambda c: (-condition_weight[c], c)):
        if label in covered:
            continue
        fits = [g for g in unique
                if label in group_conditions[g] and eligible(g) and acc + group_weight[g] <= cap]
        if not fits:
            unmet.append(label)
            continue
        pick = min(fits, key=lambda g: (rank[g], group_weight[g], g))
        take(pick)
        covered |= group_conditions[pick]

    rng = np.random.default_rng(seed)
    shuffled = [unique[k] for k in rng.permutation(len(unique))]
    for g in sorted(shuffled, key=lambda g: (rank[g], shuffled.index(g))):
        if acc >= target:
            break
        if not eligible(g) or acc + group_weight[g] > cap:
            continue
        if abs(acc + group_weight[g] - target) < abs(acc - target):
            take(g)
            covered |= group_conditions[g]
    if not chosen:
        fits = [g for g in unique if eligible(g)]
        if fits:
            pick = min(fits, key=lambda g: (abs(group_weight[g] - target), rank[g], g))
            take(pick)
            covered |= group_conditions[pick]
            info["warning"] = ("no group fits the size cap; took the closest eligible group "
                              "(val is larger than fraction × (1 + size_tolerance))")
        else:
            info["warning"] = ("every group owns a condition that no other group provides; "
                               "val would be out of distribution, so no val was generated")

    val_groups = set(chosen)
    val = [i for i in ids if key[i] in val_groups]
    train = [i for i in ids if key[i] not in val_groups]
    train_conditions = {c for i in train for c in conditions.get(i, ())}
    val_conditions = {c for i in val for c in conditions.get(i, ())}
    info.update(
        val_groups=sorted(val_groups),
        val_fraction_actual=sum(w[i] for i in val) / max(total, 1e-12),
        size_cap_fraction=fraction * (1.0 + tolerance),
        avoided_groups=sorted(g for g in val_groups if g in avoid),
        conditions_in_val=sorted(val_conditions),
        conditions_not_in_val=sorted(train_conditions - val_conditions),
        conditions_without_val_group=sorted(set(unmet) - covered),
        conditions_not_in_train=sorted(val_conditions - train_conditions),
    )
    return train, val, info


def group_holdout(
    ids: Iterable[str],
    groups: Mapping[str, str],
    fraction: float = 0.1,
    seed: int = 0,
    weights: Optional[Mapping[str, float]] = None,
) -> tuple:
    """按组抽取约 ``fraction`` 的样本（按 ``weights`` 加权，默认每条序列权重 1）作为 val。

    组按名字排序后用 ``np.random.default_rng(seed)`` 打乱，依次加入 val，直到权重达到目标；
    至少留一个组给 train。只有一个组时退化为按序列抽取并在返回信息中警告。
    返回 ``(train_ids, val_ids, info)``。
    """
    ids = list(ids)
    if not ids:
        return [], [], {"method": "group_holdout", "warning": "no sequences"}
    w = {i: float(weights.get(i, 1.0)) if weights else 1.0 for i in ids}
    key = {i: str(groups.get(i, i)) for i in ids}
    unique = sorted(set(key.values()))
    rng = np.random.default_rng(seed)
    info = {"method": "group_holdout", "key": "group_id", "fraction": fraction, "seed": seed,
            "weighting": "duration" if weights else "count"}
    target = fraction * sum(w.values())
    if len(unique) < 2:
        order = [ids[k] for k in rng.permutation(len(ids))]
        val, acc = [], 0.0
        for i in order[:-1]:
            if acc >= target:
                break
            val.append(i)
            acc += w[i]
        info["warning"] = "only one group: fell back to a sequence-level split (leakage possible)"
    else:
        group_weight = {g: 0.0 for g in unique}
        for i in ids:
            group_weight[key[i]] += w[i]
        shuffled = [unique[k] for k in rng.permutation(len(unique))]
        chosen, acc = set(), 0.0
        for g in shuffled:
            if acc >= target or len(chosen) >= len(unique) - 1:
                break
            # 只加入能让累计权重更接近目标的组，避免一个大组把 val 撑得过大
            if abs(acc + group_weight[g] - target) <= abs(acc - target):
                chosen.add(g)
                acc += group_weight[g]
        if not chosen:
            chosen.add(min(shuffled, key=lambda g: abs(group_weight[g] - target)))
        val = [i for i in ids if key[i] in chosen]
        info["val_groups"] = sorted(chosen)
    val_set = set(val)
    train = [i for i in ids if i not in val_set]
    val = [i for i in ids if i in val_set]
    info["val_fraction_actual"] = sum(w[i] for i in val) / max(sum(w.values()), 1e-12)
    return train, val, info


def _family_leakage(splits: Mapping[str, Iterable[str]], groups: Mapping[str, str], *,
                    prefix: str = "", strict_groups: bool = False) -> dict:
    """一个划分族内部的泄漏检查；``splits`` 的键不含族前缀，输出的名字带上 ``prefix``。"""
    sets = {k: set(v) for k, v in splits.items()}
    names = list(sets)
    pairs = [(a, b) for a, b in combinations(names, 2)
             if not (a.startswith("test") and b.startswith("test"))
             and (a in MAIN_SPLITS or b in MAIN_SPLITS)]
    report, leaks = [], []
    for a, b in pairs:
        seq_overlap = sorted(sets[a] & sets[b])
        ga = {groups.get(i, i) for i in sets[a]}
        gb = {groups.get(i, i) for i in sets[b]}
        group_overlap = sorted(ga & gb)
        pa, pb = prefix + a, prefix + b
        report.append({"splits": [pa, pb], "sequence_overlap": seq_overlap,
                       "group_overlap": group_overlap})
        if seq_overlap:
            leaks.append(f"{pa}/{pb} share {len(seq_overlap)} sequences")
        unseen = any(tok in n for n in (a, b) for tok in UNSEEN_TOKENS)
        if group_overlap and unseen:
            leaks.append(f"{pa}/{pb} share {len(group_overlap)} groups although one is 'unseen'")
        elif group_overlap and strict_groups and a in MAIN_SPLITS and b in MAIN_SPLITS:
            leaks.append(f"{pa}/{pb} share {len(group_overlap)} groups")
    return {"ok": not leaks, "leaks": leaks, "pairs": report}


def check_leakage(splits: Mapping[str, Iterable[str]], groups: Mapping[str, str]) -> dict:
    """报告划分之间的 ``sequence_id`` 与 ``group_id`` 重叠。

    默认族：检查主划分（train/val/test）两两之间，以及每个子集与 train/val 之间；
    ``test_*`` 子集与 ``test`` 的重叠是预期的（子集关系），不检查。
    ``leaks`` 为严重问题：主划分间序列重叠、train/val 与 unseen 子集的组重叠；
    其余组重叠（例如 seen-subject 设定）只出现在 ``pairs`` 中，由调用方作为警告报告。
    ``ok`` 只反映默认族。

    若存在 ``official_*`` 划分（转换器声明的泄漏官方划分），在 ``official`` 键下单独报告，
    并且主划分之间的任何组重叠都计为泄漏（这正是它们被降级保留的原因）。
    """
    default = {k: v for k, v in splits.items() if not split_family(k)}
    official = {k[len(OFFICIAL_PREFIX):]: v for k, v in splits.items() if split_family(k)}
    report = _family_leakage(default, groups)
    if official:
        report["official"] = _family_leakage(official, groups, prefix=OFFICIAL_PREFIX,
                                             strict_groups=True)
    return report


def condition_coverage(
    splits: Mapping[str, Iterable[str]],
    conditions: Mapping[str, Sequence[str]],
    reference: str = "train",
) -> dict:
    """各划分相对 ``reference``（默认 train）的条件覆盖审计。

    每个划分给出 ``novel``（该划分里出现、``reference`` 里完全没有的条件 → 该条件下的评测是
    分布外外推，不可学）与 ``missing``（``reference`` 里有、该划分里没有的条件）。
    ``official_*`` 族与 ``reference`` 自身不参与比较。
    """
    ref = {c for i in splits.get(reference, ()) for c in conditions.get(i, ())}
    out: dict = {"reference": reference, "reference_conditions": sorted(ref), "splits": {}}
    for name in sorted(splits):
        if name == reference or split_family(name):
            continue
        counts: dict = {}
        for i in splits[name]:
            for label in conditions.get(i, ()):
                counts[label] = counts.get(label, 0) + 1
        out["splits"][name] = {
            "novel": {k: v for k, v in sorted(counts.items()) if k not in ref},
            "missing": sorted(ref - set(counts)),
        }
    return out


def ood_subsets(
    splits: Mapping[str, Iterable[str]],
    conditions: Mapping[str, Sequence[str]],
    reference: str = "train",
    source: str = "test",
) -> dict:
    """把 ``source`` 划分里条件不在 ``reference`` 中的序列按维度拆成附加子集。

    返回 ``{f"{OOD_PREFIX}<dim>": [sequence_id]}``。主 ``test`` 不被改动：这些子集只是
    “官方 test 里有哪些条件 train 根本没有”的显式清单，供报告时单列（DESIGN 2.4）。
    """
    ref = {c for i in splits.get(reference, ()) for c in conditions.get(i, ())}
    out: dict = {}
    for i in splits.get(source, ()):
        for label in conditions.get(i, ()):
            if label not in ref:
                out.setdefault(OOD_PREFIX + _condition_dim(label), set()).add(i)
    return {k: sorted(v) for k, v in sorted(out.items())}


def resolve_splits(
    official: Mapping[str, Iterable[str]],
    available: Iterable[str],
    groups: Mapping[str, str],
    *,
    fraction: float = 0.1,
    seed: int = 0,
    weights: Optional[Mapping[str, float]] = None,
    label: str = "official",
    conditions: Optional[Mapping[str, Sequence[str]]] = None,
    coverage_only: Iterable[str] = (),
    tolerance: float = DEFAULT_SIZE_TOLERANCE,
) -> tuple:
    """合并官方划分与实际可用序列；缺 val 时从 train 分组抽取。

    返回 ``(splits, method, missing)``：``method`` 记录每个划分的来源（``label``），
    ``missing`` 为列表中存在但未成功转换的序列。``official`` 也可以是转换器给出的分组划分
    （此时 ``label`` 应注明来源）。

    给出 ``conditions`` 时用 ``stratified_group_holdout``（条件覆盖 + 分层，DESIGN 2.4），
    否则退回只看组的 ``group_holdout``（旧行为，仍可用于对照）。
    """
    avail = set(available)
    splits, missing, method = {}, {}, {}
    for name, ids in official.items():
        ids = list(dict.fromkeys(ids))  # 去重但保持顺序
        splits[name] = sorted(i for i in ids if i in avail)
        lost = sorted(i for i in ids if i not in avail)
        if lost:
            missing[name] = lost
        method[name] = label
    if "val" not in splits and "train" in splits:
        if conditions is None:
            train, val, info = group_holdout(splits["train"], groups, fraction, seed, weights)
        else:
            # val 组不与主 test 共享 group_id 时选模信号更干净，作为候选排序的首选项
            avoid = {str(groups.get(i, i)) for i in splits.get("test", ())}
            train, val, info = stratified_group_holdout(
                splits["train"], groups, conditions, fraction, seed, weights,
                tolerance=tolerance, coverage_only=coverage_only, avoid_groups=avoid)
        splits["train"], splits["val"] = sorted(train), sorted(val)
        method["train"] = f"{label} minus generated val"
        method["val"] = info
    return splits, method, missing
