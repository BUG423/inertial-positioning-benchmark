"""划分策略（DESIGN 2.4）：条件维度、分层分组抽样、条件覆盖审计与分布外子集。

这些测试只用合成元信息（不需要 h5 文件），检查四件事：不泄漏（组不跨划分）、
条件覆盖（val 的条件必须在 train 出现）、确定性（同种子同结果）与边界情况
（单组、条件只有一个组、所有组都超出体积上限）。
"""

from __future__ import annotations

import pytest

from inertial_benchmark.data.splits import (
    DEFAULT_STRATA,
    OOD_PREFIX,
    build_conditions,
    condition_coverage,
    group_holdout,
    normalise_strata,
    ood_subsets,
    resolve_splits,
    stratified_group_holdout,
)


def ridi_like() -> tuple:
    """RIDI 的结构：每个受试者一组，每组几种携带方式；只有 ``solo`` 有 ``bag_low``。"""
    plan = {
        "alice": ["handheld", "bag", "body", "leg"],
        "bob": ["handheld", "bag", "body", "leg"],
        "carol": ["handheld", "bag", "body"],
        "dave": ["handheld", "body", "leg"],
        "solo": ["handheld", "bag_low", "body"],
    }
    meta, groups = {}, {}
    for subject, placements in plan.items():
        for k, placement in enumerate(placements):
            sid = f"{subject}_{placement}{k + 1}"
            meta[sid] = {"placement": placement, "device_id": "phone",
                         "position_source": "vio", "duration_s": 100.0, "distance_m": 110.0}
            groups[sid] = subject
    ids = sorted(meta)
    weights = {i: meta[i]["duration_s"] for i in ids}
    return ids, groups, meta, weights


def conditions_of(ids, meta, strata=None, groups=None):
    conditions, info = build_conditions(ids, meta, strata, groups)
    return conditions, info


def test_normalise_strata_accepts_the_three_forms_and_is_idempotent():
    dims = normalise_strata([
        "placement",
        {"attr": "oxiod_scene", "name": "scene"},
        {"name": "variant", "of": "sequence_id", "pattern": r"^[^_]+_(.+?)\d*$"},
        {"name": "speed_bin", "of": "mean_speed", "quantiles": 3},
    ])
    assert [d["kind"] for d in dims] == ["attr", "attr", "regex", "quantile"]
    assert [d["name"] for d in dims] == ["placement", "scene", "variant", "speed_bin"]
    assert normalise_strata(dims) == dims  # 可重复调用
    assert [d["name"] for d in normalise_strata()] == list(DEFAULT_STRATA)
    with pytest.raises(ValueError, match="duplicate stratum"):
        normalise_strata(["placement", {"attr": "placement"}])
    with pytest.raises(ValueError, match="needs one of"):
        normalise_strata([{"name": "x"}])
    with pytest.raises(ValueError, match="string or a mapping"):
        normalise_strata([3])


def test_build_conditions_labels_attrs_regexes_and_quantiles():
    meta = {
        "ma_bag_low2": {"placement": "bag", "duration_s": 100.0, "distance_m": 80.0},
        "hao_bag1": {"placement": "bag", "duration_s": 100.0, "distance_m": 110.0},
        "hang_leg_new1": {"duration_s": 100.0, "distance_m": 140.0},
    }
    strata = [
        "placement",
        {"name": "variant", "of": "sequence_id", "pattern": r"^[^_]+_(.+?)\d*$"},
        {"name": "speed_bin", "of": "mean_speed", "quantiles": 3},
    ]
    conditions, info = build_conditions(sorted(meta), meta, strata)
    assert conditions["ma_bag_low2"][:2] == ("placement=bag", "variant=bag_low")
    assert conditions["hao_bag1"][:2] == ("placement=bag", "variant=bag")
    # 缺失的属性记 unknown，不报错
    assert conditions["hang_leg_new1"][0] == "placement=unknown"
    assert conditions["hang_leg_new1"][1] == "variant=leg_new"
    bins = {conditions[i][2] for i in conditions}
    assert len(bins) == 3, bins  # 三条序列速度不同 → 三个分位区间
    assert info["coverage_only"] == ["speed_bin"]
    assert info["values"]["variant"] == ["bag", "bag_low", "leg_new"]


def test_build_conditions_drops_dimensions_equivalent_to_group_id():
    # TLIO 的 device_id 就是 group_id：这种维度在分组划分下必然“val 有 train 没有”，应丢弃
    meta = {f"s{k}": {"device_id": f"headset{k}", "placement": "head"} for k in range(4)}
    groups = {f"s{k}": f"headset{k}" for k in range(4)}
    conditions, info = build_conditions(sorted(meta), meta, ["placement", "device_id"], groups)
    assert [d["name"] for d in info["strata"]] == ["placement"]
    assert "device_id" in info["dropped"]
    assert conditions["s0"] == ("placement=head",)


def test_stratified_holdout_never_puts_a_single_group_condition_in_val():
    ids, groups, meta, weights = ridi_like()
    conditions, info = conditions_of(ids, meta, ["placement"], groups)
    train, val, report = stratified_group_holdout(ids, groups, conditions, 0.2, 0, weights)
    assert val, report
    # 1) 不泄漏：组不跨划分
    assert not {groups[i] for i in train} & {groups[i] for i in val}
    assert sorted(train + val) == ids
    # 2) 条件覆盖：val 的条件都在 train 里
    train_conditions = {c for i in train for c in conditions[i]}
    assert {c for i in val for c in conditions[i]} <= train_conditions
    assert report["conditions_not_in_train"] == []
    # 3) 独占 bag_low 的组被排除，bag_low 留在 train
    assert "solo" not in report["val_groups"]
    assert report["ineligible_groups"]["solo"] == ["placement=bag_low"]
    assert "placement=bag_low" in train_conditions
    assert info["coverage_only"] == []


def test_stratified_holdout_covers_several_placements_and_is_deterministic():
    ids, groups, meta, weights = ridi_like()
    conditions, _ = conditions_of(ids, meta, ["placement"], groups)
    first = stratified_group_holdout(ids, groups, conditions, 0.2, 0, weights)
    again = stratified_group_holdout(ids, groups, conditions, 0.2, 0, weights)
    assert first[1] == again[1] and first[2]["val_groups"] == again[2]["val_groups"]
    placements = {conditions[i][0] for i in first[1]}
    assert len(placements) >= 3, placements
    for seed in range(5):  # 换种子仍然满足两条底线
        train, val, report = stratified_group_holdout(ids, groups, conditions, 0.2, seed, weights)
        assert not {groups[i] for i in train} & {groups[i] for i in val}
        assert report["conditions_not_in_train"] == []


def test_stratified_holdout_respects_the_size_cap_and_avoids_test_groups():
    ids, groups, meta, weights = ridi_like()
    conditions, _ = conditions_of(ids, meta, ["placement"], groups)
    _, val, report = stratified_group_holdout(ids, groups, conditions, 0.2, 0, weights,
                                              tolerance=0.5)
    assert report["val_fraction_actual"] <= 0.2 * 1.5 + 1e-9
    assert report["size_cap_fraction"] == pytest.approx(0.3)
    # 只有 alice 不在 test 里 → 优先抽 alice
    others = {groups[i] for i in ids} - {"alice"}
    _, val2, report2 = stratified_group_holdout(ids, groups, conditions, 0.2, 0, weights,
                                                avoid_groups=others)
    assert report2["val_groups"] == ["alice"]
    assert report2["avoided_groups"] == []


def test_stratified_holdout_boundary_cases():
    ids, groups, meta, weights = ridi_like()
    # a) 只有一个组 → 退化为按序列抽取，并警告可能泄漏
    one = {i: "only" for i in ids}
    conditions, _ = conditions_of(ids, meta, ["placement"], one)
    train, val, report = stratified_group_holdout(ids, groups=one, conditions=conditions,
                                                  fraction=0.2, weights=weights)
    assert val and "only one group" in report["warning"]
    # b) 每个组都独占一个条件 → 不生成 val，并明确说明原因
    unique = {i: {"placement": groups[i]} for i in ids}
    conditions, _ = conditions_of(ids, unique, ["placement"])
    train, val, report = stratified_group_holdout(ids, groups, conditions, 0.2, 0, weights)
    assert val == [] and train == ids
    assert "out of distribution" in report["warning"]
    # c) 组都比上限大 → 取最接近目标的合格组并警告超额（IMUNet 只有 4 名受试者的情形）
    conditions, _ = conditions_of(ids, meta, ["placement"], groups)
    _, val, report = stratified_group_holdout(ids, groups, conditions, 0.02, 0, weights)
    assert val and "size cap" in report["warning"]
    assert report["val_fraction_actual"] > 0.03
    # d) 空输入
    assert stratified_group_holdout([], {}, {})[2]["warning"] == "no sequences"


def test_coverage_only_dimensions_do_not_gate_eligibility():
    ids, groups, meta, weights = ridi_like()
    # 给每个组一个独占的速度分位：作为硬条件会让所有组失去资格，作为覆盖维度则不应有影响
    for k, sid in enumerate(ids):
        meta[sid] = {**meta[sid], "distance_m": 100.0 + 10.0 * k}
    strata = ["placement", {"name": "speed_bin", "of": "mean_speed", "quantiles": 3}]
    conditions, info = conditions_of(ids, meta, strata, groups)
    _, val, report = stratified_group_holdout(ids, groups, conditions, 0.2, 0, weights,
                                              coverage_only=info["coverage_only"])
    assert val
    hard = {c for i in val for c in conditions[i] if not c.startswith("speed_bin=")}
    assert hard <= {c for i in ids if i not in val for c in conditions[i]}


def test_condition_coverage_and_ood_subsets_flag_conditions_absent_from_train():
    meta = {
        "tr0": {"placement": "handheld", "device_id": "asus4"},
        "tr1": {"placement": "bag", "device_id": "asus4"},
        "va0": {"placement": "handheld", "device_id": "asus4"},
        "te0": {"placement": "handheld", "device_id": "asus6"},  # 训练里没有这台设备
        "te1": {"placement": "trolley", "device_id": "asus4"},   # 训练里没有这种携带方式
    }
    splits = {"train": ["tr0", "tr1"], "val": ["va0"], "test": ["te0", "te1"],
              "official_train": ["te0"]}
    conditions, _ = build_conditions(sorted(meta), meta, ["placement", "device_id"])
    coverage = condition_coverage(splits, conditions)
    assert coverage["splits"]["val"]["novel"] == {}
    assert coverage["splits"]["val"]["missing"] == ["placement=bag"]
    assert coverage["splits"]["test"]["novel"] == {"device_id=asus6": 1, "placement=trolley": 1}
    assert "official_train" not in coverage["splits"]  # 官方泄漏族不参与比较
    assert ood_subsets(splits, conditions) == {f"{OOD_PREFIX}device_id": ["te0"],
                                               f"{OOD_PREFIX}placement": ["te1"]}


def test_resolve_splits_uses_the_stratified_holdout_only_with_conditions():
    ids, groups, meta, weights = ridi_like()
    official = {"train": ids, "test": []}
    conditions, _ = conditions_of(ids, meta, ["placement"], groups)
    splits, method, _ = resolve_splits(official, ids, groups, fraction=0.2, weights=weights,
                                       conditions=conditions)
    assert method["val"]["method"] == "stratified_group_holdout"
    assert "solo" not in method["val"]["val_groups"]
    legacy, legacy_method, _ = resolve_splits(official, ids, groups, fraction=0.2,
                                              weights=weights)
    assert legacy_method["val"]["method"] == "group_holdout"  # 旧行为仍可用
    assert set(legacy["train"]) | set(legacy["val"]) == set(ids)


def test_group_holdout_keeps_its_old_behaviour():
    ids, groups, meta, weights = ridi_like()
    train, val, info = group_holdout(ids, groups, 0.2, 0, weights)
    assert info["method"] == "group_holdout"
    assert not {groups[i] for i in train} & {groups[i] for i in val}
    assert sorted(train + val) == ids
