import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))

from hrm_blueprint import (
    ModelConfig,
    TransformerConfig,
    HRMConfig,
    EngineConfig,
    HybridHRMLLMBlueprint,
    EngineBridge,
)


def test_blueprint_construction():
    config = ModelConfig(
        vocab_size=8000,
        transformer=TransformerConfig(num_layers=2, hidden_dim=128, intermediate_dim=384),
        hrm=HRMConfig(hidden_dim=128, max_reasoning_steps=4),
    )
    blueprint = HybridHRMLLMBlueprint.create(config)

    assert blueprint.name == "HRM-LLM-Hybrid-Base"
    assert len(blueprint.layers) > 0
    assert blueprint.total_parameters > 0

    # Verify presence of both transformer layers and hrm modules
    layer_names = [l.name for l in blueprint.layers]
    assert "token_embeddings" in layer_names
    assert "layer_0.attention" in layer_names
    assert "hrm.low_level_executor" in layer_names
    assert "hrm.high_level_planner" in layer_names
    assert "hrm.act_qhead" in layer_names
    assert "lm_decoder_head" in layer_names


def test_blueprint_modification():
    config = ModelConfig(
        vocab_size=4000,
        hrm=HRMConfig(hidden_dim=64, max_reasoning_steps=5),
    )
    blueprint = HybridHRMLLMBlueprint.create(config)

    HybridHRMLLMBlueprint.modify_reasoning_depth(blueprint, new_max_steps=12)

    planner_layer = next(l for l in blueprint.layers if l.name == "hrm.high_level_planner")
    assert planner_layer.hyperparams["max_steps"] == 12


def test_native_c_engine_bridge():
    bridge = EngineBridge()
    bridge.init_engine(arena_mb=32, threads=2)
    assert "HRM-LLM Engine" in bridge.get_version()

    D = 64
    B = 1

    np_zh = np.ones((B, D), dtype=np.float32) * 0.1
    np_zl = np.ones((B, D), dtype=np.float32) * 0.1
    np_input = np.ones((B, D), dtype=np.float32) * 0.2

    np_w_low = np.zeros((3 * D, D), dtype=np.float32)
    np_u_low = np.zeros((3 * D, D), dtype=np.float32)
    np_w_high = np.zeros((3 * D, D), dtype=np.float32)
    np_u_high = np.zeros((3 * D, D), dtype=np.float32)

    np_w_q = np.zeros((2, D), dtype=np.float32)
    np_b_q = np.array([2.0, -2.0], dtype=np.float32)  # Force early halt

    t_zh = bridge.wrap_numpy(np_zh)
    t_zl = bridge.wrap_numpy(np_zl)
    t_input = bridge.wrap_numpy(np_input)
    t_w_low = bridge.wrap_numpy(np_w_low)
    t_u_low = bridge.wrap_numpy(np_u_low)
    t_w_high = bridge.wrap_numpy(np_w_high)
    t_u_high = bridge.wrap_numpy(np_u_high)
    t_w_q = bridge.wrap_numpy(np_w_q)
    t_b_q = bridge.wrap_numpy(np_b_q)

    result = bridge.run_reasoning_loop(
        z_h=t_zh,
        z_l=t_zl,
        input_emb=t_input,
        w_low=t_w_low,
        u_low=t_u_low,
        b_low=None,
        w_high=t_w_high,
        u_high=t_u_high,
        b_high=None,
        w_qhead=t_w_q,
        b_qhead=t_b_q,
        max_steps=5,
        low_steps=2,
        act_threshold=0.5,
    )

    assert result.steps_executed == 1
    assert result.halted_early == 1
    assert result.halt_confidence > 0.5

    bridge.reset_arena()
    bridge.close()
