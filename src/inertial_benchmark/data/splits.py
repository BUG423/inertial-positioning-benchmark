"""数据划分（DESIGN 2.4）：官方划分优先、按 ``group_id`` 分组抽取 val、泄漏检查。

划分文件分两个“族”：默认族（``train/val/test`` 与 ``test_*`` 子集，训练与评测使用）和
官方泄漏族（``official_*``，仅当转换器声明官方划分存在泄漏时写出，只用于与文献对照）。
泄漏检查在各族内部分别进行，两族之间的重叠是预期的，不检查。
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Iterable, Mapping, Optional, Union

import numpy as np

PathLike = Union[str, Path]
MAIN_SPLITS = ("train", "val", "test")
# 名字含这些词的官方子集表示“未见过的受试者/场景”，与训练集共享 group 视为泄漏
UNSEEN_TOKENS = ("unseen", "unknown", "novel")
# 转换器声明官方划分泄漏时，官方划分以该前缀另存（DESIGN 2.4）
OFFICIAL_PREFIX = "official_"


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


def resolve_splits(
    official: Mapping[str, Iterable[str]],
    available: Iterable[str],
    groups: Mapping[str, str],
    *,
    fraction: float = 0.1,
    seed: int = 0,
    weights: Optional[Mapping[str, float]] = None,
    label: str = "official",
) -> tuple:
    """合并官方划分与实际可用序列；缺 val 时从 train 分组抽取。

    返回 ``(splits, method, missing)``：``method`` 记录每个划分的来源（``label``），
    ``missing`` 为列表中存在但未成功转换的序列。``official`` 也可以是转换器给出的分组划分
    （此时 ``label`` 应注明来源）。
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
        train, val, info = group_holdout(splits["train"], groups, fraction, seed, weights)
        splits["train"], splits["val"] = sorted(train), sorted(val)
        method["train"] = f"{label} minus generated val"
        method["val"] = info
    return splits, method, missing
