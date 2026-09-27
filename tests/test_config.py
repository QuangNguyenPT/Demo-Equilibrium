import json

import pytest

from hrm.config import ConfigError, ModelConfig, SubstrateConfig


def test_defaults_validate():
    ModelConfig().validate()


def test_save_load_round_trip(tmp_path):
    cfg = ModelConfig(substrate=SubstrateConfig(dim=1024, seed=7))
    path = tmp_path / "config.json"
    cfg.save(path)
    assert ModelConfig.load(path) == cfg


@pytest.mark.parametrize("bad", [
    {"dim": 1000},        # not a power of two
    {"dim": 32},          # below minimum
    {"seed": -1},
    {"dtype": "float32"},
    {"codebook_cache_mb": 0},
])
def test_out_of_range_rejected(bad):
    with pytest.raises(ConfigError):
        ModelConfig(substrate=SubstrateConfig(**bad)).validate()


def test_unknown_field_rejected():
    data = ModelConfig().to_dict()
    data["substrate"]["dimm"] = 1024
    with pytest.raises(ConfigError, match="unknown"):
        ModelConfig.from_dict(data)


def test_missing_field_rejected_not_defaulted():
    data = ModelConfig().to_dict()
    del data["substrate"]["seed"]
    with pytest.raises(ConfigError, match="missing"):
        ModelConfig.from_dict(data)


@pytest.mark.parametrize("field,value", [("dim", True), ("dim", "4096"), ("dim", 4096.0),
                                         ("dtype", 64)])
def test_wrong_type_rejected(field, value):
    data = ModelConfig().to_dict()
    data["substrate"][field] = value
    with pytest.raises(ConfigError, match="expected"):
        ModelConfig.from_dict(data)


def test_invalid_json_reported(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid JSON"):
        ModelConfig.load(path)


def test_saved_file_is_plain_json(tmp_path):
    path = tmp_path / "config.json"
    ModelConfig().save(path)
    assert json.loads(path.read_text(encoding="utf-8"))["substrate"]["dim"] == 4096
