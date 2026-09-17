import pytest

from inertial_benchmark.cfg import (
    NULLABLE_KEYS,
    ConfigError,
    get_cfg,
    load_default,
    load_model_cfg,
    parse_key_value,
    parse_str_list,
    parse_value,
)
from inertial_benchmark.data.manifest import dataset_yaml_names, resolve_dataset
from inertial_benchmark.metrics import fitness_keys
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


def test_none_is_only_allowed_on_nullable_keys():
    defaults = load_default()
    empty = {k for k, v in defaults.items() if v is None}
    assert empty == set(NULLABLE_KEYS)  # 可空键必须与 default.yaml 中默认为空的键一致
    for key in NULLABLE_KEYS:
        assert getattr(get_cfg({key: None}), key) is None
    for key in ("epochs", "batch", "lr", "window", "fitness", "augment", "deterministic"):
        with pytest.raises(ConfigError, match="not allowed"):
            get_cfg({key: None})


def test_fitness_is_validated_when_the_config_is_parsed():
    assert get_cfg({"fitness": "t_rte_1s"}).fitness == "t_rte_1s"
    assert get_cfg({"fitness": "d_rte_5m", "d_rte": [5.0]}).fitness == "d_rte_5m"
    assert "ate" in fitness_keys() and "params" not in fitness_keys()
    assert "ate_oracle" not in fitness_keys()  # 与模型无关，不能用于选模
    with pytest.raises(ConfigError, match="did you mean ate"):
        get_cfg({"fitness": "ate_"})
    with pytest.raises(ConfigError, match="lower-is-better"):
        get_cfg({"fitness": "plr"})  # 越接近 1 越好，不是越小越好
    with pytest.raises(ConfigError, match="lower-is-better"):
        get_cfg({"fitness": "d_rte_5m"})  # d_rte 配置里没有 5 m


def test_sequence_ids_are_parsed_as_strings():
    assert parse_key_value(["only=[010]"]) == {"only": ["010"]}  # 不是八进制 8
    assert parse_key_value(["only=007"]) == {"only": ["007"]}  # 不是整数 7
    assert parse_key_value(["only=a, b ,'c'"]) == {"only": ["a", "b", "c"]}
    assert parse_str_list("[x]") == ["x"]
    for text in ("only=[]", "only=[a", "only=a]", "only=[a,,b]", "only={a: 1}"):
        with pytest.raises(ConfigError):
            parse_key_value([text])


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
