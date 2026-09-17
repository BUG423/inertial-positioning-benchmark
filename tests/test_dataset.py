from types import SimpleNamespace

import numpy as np
import pytest
from synthetic import make_sequence

torch = pytest.importorskip("torch")

from inertial_benchmark.data.build import build_dataloader, build_dataset  # noqa: E402
from inertial_benchmark.data.dataset import InertialDataset  # noqa: E402
from inertial_benchmark.data.format import save_sequence  # noqa: E402
from inertial_benchmark.data.splits import write_split  # noqa: E402
from inertial_benchmark.data.views import ViewConfig  # noqa: E402


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    root = tmp_path_factory.mktemp("ds")
    paths = []
    for k in range(3):
        seq = make_sequence(duration=8.0 + k, seed=10 + k, sequence_id=f"s{k}",
                            group_id=f"g{k}", device_yaw_offset=0.2 * k)
        if k == 1:
            seq.valid_imu[300:320] = False
        paths.append(save_sequence(root / "sequences" / f"s{k}.h5", seq))
    write_split(root / "splits" / "train.txt", ["s0", "s1"])
    write_split(root / "splits" / "val.txt", ["s2"])
    return root, paths


VIEWS = [
    ViewConfig(dims=2),
    ViewConfig(dims=3, remove_gravity=True, target="displacement"),
    ViewConfig(frame="gravity_yaw_local", dims=2, target="velocity_at_end"),
    ViewConfig(frame="body", dims=3),
    ViewConfig(orientation="device", dims=2),
]


@pytest.mark.parametrize("view", VIEWS, ids=lambda v: f"{v.frame}-{v.orientation}-{v.target}")
def test_lazy_matches_cached(files, view):
    _, paths = files
    cached = InertialDataset(paths, view, stride=37)
    lazy = InertialDataset(paths, view, stride=37, cache=False)
    assert len(cached) == len(lazy) > 0
    np.testing.assert_array_equal(cached.index, lazy.index)
    for i in range(0, len(cached), 7):
        a, b = cached[i], lazy[i]
        assert a["seq"] == b["seq"] and a["start"] == b["start"]
        np.testing.assert_allclose(a["imu"].numpy(), b["imu"].numpy(), atol=1e-6)
        np.testing.assert_allclose(a["target"].numpy(), b["target"].numpy(), atol=1e-6)


def test_index_skips_invalid_and_short(files, tmp_path):
    _, paths = files
    ds = InertialDataset(paths, ViewConfig(), stride=10)
    s1 = ds.index[ds.index[:, 0] == 1, 1]
    assert not np.any((s1 < 320) & (s1 + 200 > 300))
    short = make_sequence(duration=0.5, sequence_id="tiny")
    ds2 = InertialDataset([short], ViewConfig(), stride=10)
    assert len(ds2) == 0


def test_augmentation_reproducible(files):
    _, paths = files
    ds = InertialDataset(paths, ViewConfig(), stride=20, training=True, seed=3,
                         augment=["time_shift", "random_yaw", {"name": "noise", "acc": 0.1}])
    a, b = ds[5], ds[5]
    np.testing.assert_array_equal(a["imu"].numpy(), b["imu"].numpy())
    ds.set_epoch(1)
    c = ds[5]
    assert not np.allclose(a["imu"].numpy(), c["imu"].numpy())
    # 平移后的窗口仍在合法范围且全部有效
    for i in range(len(ds)):
        item = ds[i]
        k, start = item["seq"], item["start"]
        assert abs(start - ds.index[i, 1]) <= 10
        assert ds.valid_masks[k][start:start + 200].all()
    # 目标范数在随机偏航下保持
    plain = InertialDataset(paths, ViewConfig(), stride=20)
    aug = InertialDataset(paths, ViewConfig(), stride=20, training=True,
                          augment=["random_yaw"])
    np.testing.assert_allclose(plain[3]["target"].norm(), aug[3]["target"].norm(), rtol=1e-5)


def test_one_sided_time_shift(files):
    _, paths = files
    ds = InertialDataset(paths, ViewConfig(), stride=10, training=True, seed=0,
                         augment=[{"time_shift": {"min_shift": 0, "max_shift": 9}}])
    assert ds.shift_range == (0, 9)
    shifts = []
    for epoch in range(3):
        ds.set_epoch(epoch)
        shifts += [ds[i]["start"] - ds.index[i, 1] for i in range(0, len(ds), 7)]
    assert min(shifts) >= 0 and max(shifts) <= 9 and len(set(shifts)) > 5


def test_dataloader_independent_of_workers(files):
    _, paths = files
    ds = InertialDataset(paths, ViewConfig(), stride=20, training=True, seed=1,
                         augment=["random_yaw", "time_shift"])
    batches = []
    for workers in (0, 2):
        loader = build_dataloader(ds, batch=16, shuffle=True, workers=workers, seed=7)
        batches.append([b["imu"] for b in loader])
    assert len(batches[0]) == len(batches[1])
    for x, y in zip(batches[0], batches[1]):
        torch.testing.assert_close(x, y)
    first = next(iter(build_dataloader(ds, 16, True, 0, seed=7)))
    assert first["imu"].shape == (16, 6, 200) and first["target"].shape == (16, 2)


def test_build_dataset_from_directory(files):
    root, _ = files
    cfg = SimpleNamespace(data=str(root), stride=10, eval_stride=20, augment=["random_yaw"],
                          cache=True, seed=0, window=200, frame="gravity_world",
                          orientation="reference", remove_gravity=False,
                          target="avg_velocity", dims=2)
    train = build_dataset(cfg, "train")
    val = build_dataset(cfg, "val")
    assert train.training and not val.training
    assert train.sequence_ids == ["s0", "s1"] and val.sequence_ids == ["s2"]
    assert val.stride == 20 and val.augs == []
    with pytest.raises(FileNotFoundError):
        build_dataset(cfg, "test")
