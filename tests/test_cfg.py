import pytest

from inertial_benchmark.cfg import (
    ConfigError,
    get_cfg,
    load_default,
    load_model_cfg,
    parse_key_value,
    parse_value,
)
from inertial_benchmark.data.manifest import dataset_yaml_names, resolve_dataset
from inertial_benchmark.utils import CFG_DIR, yaml_load

REQUIRED_DEFAULT_KEYS = (
    "mode model data epochs batch lr optimizer scheduler weight_decay patience seed "
    "deterministic device workers amp window stride eval_stride frame orientation "
    "remove_gravity target dims augment fitness val_interval project name exist_ok "
    "save_predictions plots recipe"
).split()


def test_default_has_all_documented_keys():
    defaults = load_default()
    missing = [k for k in REQUIRED_DEFAULT_KEYS if k not in defaults]
    assert not missing


def test_model_input_and_recipe_merging():
    cfg = get_cfg({"model": "ronin_resnet18"})
    assert cfg.window == 200 and cfg.frame == "gravity_world" and cfg.dims == 2
    assert cfg.recipe == "unified" and cfg.optimizer == "adamw"
    official = get_cfg({"model": "ronin_resnet18", "recipe": "official"})
    assert official.lr == pytest.approx(1e-4) and official.optimizer == "adam"
    assert official.scheduler == "plateau" and official.batch == 128
    # 用户覆盖优先于配方
    user = get_cfg({"model": "ronin_resnet18", "recipe": "official", "lr": 3e-4})
    assert user.lr == pytest.approx(3e-4)


def test_unknown_keys_and_suggestions():
    with pytest.raises(ConfigError, match="did you mean epochs"):
        get_cfg({"epoch": 3})
    with pytest.raises(ConfigError, match="no recipe"):
        get_cfg({"model": {"name": "x", "arch": "ronin_resnet"}, "recipe": "official"})


@pytest.mark.parametrize("key,value,expected", [
    ("lr", 1, 1.0), ("epochs", 2.0, 2), ("resume", "runs/x/last.pt", "runs/x/last.pt"),
    ("device", 0, 0), ("name", 7, "7"), ("augment", ("random_yaw",), ["random_yaw"]),
])
def test_type_coercion(key, value, expected):
    assert getattr(get_cfg({key: value}), key) == expected


@pytest.mark.parametrize("override", [
    {"epochs": "ten"}, {"amp": "yes"}, {"lr": True}, {"frame": "enu"}, {"dims": 4},
    {"frame": "body", "dims": 2}, {"epochs": 0}, {"augment": "random_yaw"},
    {"model_args": [1]}, {"optimizer": "lamb"},
])
def test_invalid_values(override):
    with pytest.raises(ConfigError):
        get_cfg(override)


def test_parse_key_value():
    args = parse_key_value([
        "epochs=2", "lr=1e-3", "amp=False", "name=exp", "device=0", "augment=[random_yaw,noise]",
        "t_rte=[1, 1e1]", "model_args={dropout: 0.1}", "pretrained=none", "--batch=4",
        "device_str=cuda:0",
    ])
    assert args == {
        "epochs": 2, "lr": 1e-3, "amp": False, "name": "exp", "device": 0,
        "augment": ["random_yaw", "noise"], "t_rte": [1, 10.0],
        "model_args": {"dropout": 0.1}, "pretrained": None, "batch": 4,
        "device_str": "cuda:0",
    }
    assert parse_value("0,1") == "0,1"
    assert parse_value("-2.5") == -2.5
    with pytest.raises(ConfigError):
        parse_key_value(["epochs"])


def test_dataset_yamls_are_complete(monkeypatch, tmp_path):
    monkeypatch.setenv("IPB_DATASETS", str(tmp_path))
    names = dataset_yaml_names()
    assert set(names) >= {"ronin", "ridi", "oxiod", "tlio", "idol", "rnin", "imunet",
                          "pedlocdata"}
    for name in names:
        raw = yaml_load(CFG_DIR / "datasets" / f"{name}.yaml")
        assert {"name", "path", "splits", "test_splits", "notes"} <= set(raw), name
        spec = resolve_dataset(name)
        assert spec.name == name
        assert spec.root == tmp_path / raw["path"]
        assert {"train", "val", "test"} <= set(spec.splits)
        assert spec.test_splits
    assert resolve_dataset("ronin").test_splits == ["test_seen", "test_unseen"]
    with pytest.raises(FileNotFoundError, match="cannot resolve"):
        resolve_dataset("no_such_dataset")


def test_model_yamls_resolve():
    for name in ("ronin_resnet18", "ronin_resnet50", "ronin_resnet101"):
        mcfg = load_model_cfg(name)
        assert mcfg["arch"] == "ronin_resnet"
        assert {"official", "unified"} <= set(mcfg["recipes"])
        assert mcfg["input"]["window"] == 200
    with pytest.raises(ConfigError, match="unknown model"):
        load_model_cfg("not_a_model")
    with pytest.raises(ConfigError, match="required"):
        load_model_cfg(None)
