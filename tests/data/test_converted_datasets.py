"""已转换数据集（IPB v1 发布，docs/DATASETS_V1.md）的验收测试；数据不存在时跳过。

根目录取环境变量 ``IPB_DATASETS``，缺省为本机的 ``/workspace/webCodex/datasets/ipb``。
这里只做清单级检查（不逐条读取，秒级完成）：``ipb check`` 必须通过、划分策略必须符合
DESIGN 2.4，并锁定发布说明中的规模与指纹。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from inertial_benchmark.data.manifest import check_dataset
from inertial_benchmark.data.splits import load_splits

ROOT = Path(os.environ.get("IPB_DATASETS", "/workspace/webCodex/datasets/ipb"))

# 数据集 → (序列数, 时长 h, 距离 km, 转换器, 指纹前 16 位)
RELEASE = {
    "ronin": (151, 23.11, 61.94, "ronin@1.0", "f29e5c58e71935cc"),
    "ridi": (94, 2.71, 10.30, "ridi@1.1", "6e47893cb3b6dbf0"),
    "oxiod": (126, 13.60, 36.90, "oxiod@1.1", "559c7c3de007887e"),
    "tlio": (354, 31.60, 37.63, "tlio@1.0.0", "c2d0ddbe9d6ea11f"),
    "idol": (130, 19.89, 64.94, "idol@1.0.0", "dadeed68bbc3f119"),
    "rnin": (300, 8.49, 25.56, "rnin@1.1.0", "6d7f93e5734c88b5"),
    "imunet": (113, 7.91, 30.80, "imunet@1.0", "3fe0f633264ab36a"),
    "pedlocdata": (1924, 89.14, 257.58, "pedlocdata@1.1.0", "79b8e288ac4be804"),
}
EXTRA_SPLITS = {
    "ridi": {"test_unseen_subject": 11, "test_unlisted_seen_subject": 11},
    "oxiod": {"test_unseen_subject": 30, "test_unseen_device": 26},
}


def dataset_dirs():
    return [name for name in RELEASE if (ROOT / name / "dataset.json").exists()]


pytestmark = pytest.mark.skipif(not dataset_dirs(),
                                reason="no converted dataset under IPB_DATASETS")


@pytest.mark.parametrize("name", sorted(RELEASE))
def test_release_manifest_and_checks(name):
    root = ROOT / name
    if not (root / "dataset.json").exists():
        pytest.skip(f"{name} not converted")
    sequences, hours, km, converter, fingerprint = RELEASE[name]
    manifest = json.loads((root / "dataset.json").read_text())
    stats = manifest["statistics"]
    assert manifest["converter"] == converter
    assert manifest["sample_rate_hz"] == 200.0
    assert stats["num_sequences"] == sequences
    assert stats["total_duration_h"] == pytest.approx(hours, abs=0.01)
    assert stats["total_distance_m"] / 1000 == pytest.approx(km, abs=0.01)
    assert manifest["fingerprint"].startswith(fingerprint)
    assert manifest["unassigned"] == []
    report = check_dataset(root)
    assert report["ok"], report["errors"]
    assert report["duplicates"] == []
    assert report["unassigned"] == []


@pytest.mark.parametrize("name", sorted(RELEASE))
def test_split_policy(name):
    root = ROOT / name
    if not (root / "dataset.json").exists():
        pytest.skip(f"{name} not converted")
    manifest = json.loads((root / "dataset.json").read_text())
    policy = manifest["split_policy"]
    splits = load_splits(root)
    assert splits["train"] and splits["val"] and splits["test"]
    official = {k for k in splits if k.startswith("official_")}
    if name == "pedlocdata":
        # 官方划分泄漏：默认用分组划分，官方划分降级保存
        assert policy["default"] == "grouped" and policy["official_splits_leak"] is True
        assert policy["official_leaks"] and "recording" in policy["reason"]
        assert official == {"official_train", "official_val", "official_test",
                            "official_test_yt", "official_test_demo"}
        assert set(splits["train"]) & set(splits["official_test"])  # 两套划分不得混用
    else:
        assert policy["default"] == "official" and not policy["official_splits_leak"]
        assert not official
    # 附加测试子集原样写出，且绝不进入 train/val
    for split, size in EXTRA_SPLITS.get(name, {}).items():
        assert len(splits[split]) == size
        assert split in policy["extra_splits"]
        assert not set(splits[split]) & (set(splits["train"]) | set(splits["val"]))
    # 每条已转换序列都属于至少一个默认划分
    assigned = {i for k, v in splits.items() if not k.startswith("official_") for i in v}
    assert set(manifest["sequences"]) <= assigned
