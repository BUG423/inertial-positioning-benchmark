"""插件钩子：``IPB_PLUGINS`` 导入外部模块，``register_augmentation`` 注册外部增强。"""

import textwrap

import numpy as np
import pytest

from inertial_benchmark.data.augment import (
    REGISTRY,
    Augmentation,
    RandomYaw,
    build_augmentations,
    register_augmentation,
)
from inertial_benchmark.utils import plugins

PLUGIN = textwrap.dedent('''
    import numpy as np
    from inertial_benchmark.data.augment import Augmentation, register_augmentation

    @register_augmentation("test_plugin_flip")
    class Flip(Augmentation):
        stage = 1

        def __init__(self, sign=-1.0):
            self.sign = float(sign)

        def __call__(self, sample, rng):
            sample["imu"] *= self.sign
            return sample

    try:
        import torch
        from inertial_benchmark.nn import BaseModel, register_model

        @register_model("test_plugin_model")
        class PluginModel(BaseModel):
            def __init__(self, input_spec):
                super().__init__(input_spec)
                self.bias = torch.nn.Parameter(torch.zeros(input_spec.dims))

            def forward(self, imu):
                return {"vel": self.bias.expand(imu.shape[0], -1)}
    except ImportError:
        pass
''')


@pytest.fixture
def plugin_file(tmp_path, monkeypatch):
    path = tmp_path / "my_plugin.py"
    path.write_text(PLUGIN)
    monkeypatch.setenv(plugins.ENV_VAR, f" {path} ,")
    yield path
    REGISTRY.pop("test_plugin_flip", None)
    plugins._LOADED.pop(str(path), None)
    try:
        from inertial_benchmark.nn import MODELS

        MODELS.pop("test_plugin_model", None)
    except ImportError:
        pass


def test_plugin_names_parsing(monkeypatch):
    monkeypatch.setenv(plugins.ENV_VAR, "a.b, c.py ,,")
    assert plugins.plugin_names() == ["a.b", "c.py"]
    assert plugins.plugin_names("") == []
    with pytest.raises(ImportError, match="cannot load plugin"):
        plugins.load_plugins(["no_such_module_for_ipb_tests"])


def test_plugin_augmentation_is_found_by_name(plugin_file):
    assert "test_plugin_flip" not in REGISTRY
    _, augs = build_augmentations(["random_yaw", {"name": "test_plugin_flip", "sign": 2.0}])
    assert len(augs) == 2 and isinstance(augs[0], RandomYaw)
    flip = augs[-1]
    assert type(flip).__name__ == "Flip" and flip.sign == 2.0
    sample = {"imu": np.ones((6, 4), np.float32), "target": np.zeros(2, np.float32)}
    out = flip(sample, np.random.default_rng(0))
    np.testing.assert_array_equal(out["imu"], 2.0)
    # 同一插件只导入一次
    assert plugins.load_plugins()[0] is plugins.load_plugins()[0]


def test_register_augmentation_checks():
    with pytest.raises(TypeError):
        register_augmentation("not_an_aug")(object)
    with pytest.raises(KeyError, match="already registered"):
        @register_augmentation("random_yaw")
        class Other(Augmentation):
            pass


def test_plugin_model_is_registered(plugin_file):
    torch = pytest.importorskip("torch")
    from inertial_benchmark.cfg import get_cfg
    from inertial_benchmark.nn import build_model, list_models

    assert "test_plugin_model" in list_models()
    model = build_model(get_cfg({"model": "test_plugin_model"}))
    assert model(torch.zeros(3, 6, 200))["vel"].shape == (3, 2)
