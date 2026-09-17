"""``tests/models`` 的共享 pytest 夹具（合成数据集）；工具函数见 ``model_testing.py``。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inertial_benchmark.data.convert import convert_dataset

FAKE = Path(__file__).resolve().parents[1] / "fake_converter.py"


@pytest.fixture(scope="session")
def synthetic_dataset(tmp_path_factory) -> Path:
    """4 条训练序列（分组 val 抽走 1 条）+ 1 条测试序列，200 Hz、每条 10 s。"""
    root = tmp_path_factory.mktemp("models_data")
    entries = [{"id": f"tr{k}", "seed": k, "group": f"g{k}", "split": "train",
                "duration": 10.0, "imu_rate": 200.0} for k in range(4)]
    entries.append({"id": "te0", "seed": 50, "group": "gx", "split": "test",
                    "duration": 10.0, "imu_rate": 200.0})
    (root / "raw").mkdir(parents=True)
    (root / "raw" / "spec.json").write_text(json.dumps({"sequences": entries}))
    convert_dataset("fake", root / "raw", root / "fake", converter=FAKE, val_fraction=0.25)
    return root / "fake"
