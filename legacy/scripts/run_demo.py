import sys
import os
import time
import numpy as np

# Ensure root and python directories are in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))

from hrm_blueprint import (
    ModelConfig,
    TransformerConfig,
    HRMConfig,
    HybridHRMLLMBlueprint,
    EngineBridge,
)


def main():
    print("=" * 70)
    print("  HRM+LLM Hybrid AI Architecture Demo")
    print("  Python Blueprint Layer -> C++ High-Performance System Engine")
    print("=" * 70)

    # 1. Define Model Blueprint Layout in Python
    print("\n[STEP 1] Generating Model Architecture Blueprint in Python...")
    config = ModelConfig(
        vocab_size=16000,
        transformer=TransformerConfig(
            num_layers=4,
            hidden_dim=256,
            num_attention_heads=4,
            intermediate_dim=768,
            max_seq_len=1024,
        ),
        hrm=HRMConfig(
            hidden_dim=256,
            max_reasoning_steps=6,
            low_level_substeps=2,
            act_threshold=0.55,
            use_act_halting=True,
        ),
    )

    blueprint = HybridHRMLLMBlueprint.create(config)
    print(blueprint.summary())

    # 2. Modify Blueprint Layout on the fly in Python
    print("\n[STEP 2] Demonstrating rapid Python blueprint modification...")
    HybridHRMLLMBlueprint.modify_reasoning_depth(blueprint, new_max_steps=10)

    # 3. Connect to Native C++ Execution Engine
    print("\n[STEP 3] Initializing Native C++ Engine via C-ABI Bridge...")
    bridge = EngineBridge()
    bridge.init_engine(arena_mb=64, threads=4)
    print(f"  -> Connected to: {bridge.get_version()}")

    # 4. Prepare Tensors with zero-copy NumPy buffers
    B, D = 1, config.transformer.hidden_dim
    print(f"\n[STEP 4] Allocating Tensors for Reasoning Loop (B={B}, D={D})...")

    np_zh = np.ones((B, D), dtype=np.float32) * 0.1
    np_zl = np.ones((B, D), dtype=np.float32) * 0.1
    np_input = np.ones((B, D), dtype=np.float32) * 0.2

    np_w_low = np.random.randn(3 * D, D).astype(np.float32) * 0.01
    np_u_low = np.random.randn(3 * D, D).astype(np.float32) * 0.01
    np_w_high = np.random.randn(3 * D, D).astype(np.float32) * 0.01
    np_u_high = np.random.randn(3 * D, D).astype(np.float32) * 0.01

    np_w_q = np.random.randn(2, D).astype(np.float32) * 0.02
    np_b_q = np.array([1.5, -1.5], dtype=np.float32)  # High halt logit -> trigger early halt

    t_zh = bridge.wrap_numpy(np_zh)
    t_zl = bridge.wrap_numpy(np_zl)
    t_input = bridge.wrap_numpy(np_input)
    t_w_low = bridge.wrap_numpy(np_w_low)
    t_u_low = bridge.wrap_numpy(np_u_low)
    t_w_high = bridge.wrap_numpy(np_w_high)
    t_u_high = bridge.wrap_numpy(np_u_high)
    t_w_q = bridge.wrap_numpy(np_w_q)
    t_b_q = bridge.wrap_numpy(np_b_q)

    # 5. Execute C++ Kernel
    print("\n[STEP 5] Executing HRM Hierarchical Reasoning Loop in C++...")
    start_t = time.perf_counter()
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
        max_steps=config.hrm.max_reasoning_steps,
        low_steps=config.hrm.low_level_substeps,
        act_threshold=config.hrm.act_threshold,
    )
    elapsed_ms = (time.perf_counter() - start_t) * 1000.0

    print(f"  -> Steps Executed: {result.steps_executed} / {config.hrm.max_reasoning_steps}")
    print(f"  -> Early Halting: {'TRIGGERED' if result.halted_early else 'NOT TRIGGERED'}")
    print(f"  -> Halt Confidence: {result.halt_confidence:.4f}")
    print(f"  -> Execution Latency: {elapsed_ms:.3f} ms")

    # Verify updated z_h in NumPy
    zh_view = t_zh.numpy_view()
    print(f"  -> Final z_h state mean: {float(np.mean(zh_view)):.6f}")

    # 6. Cleanup
    bridge.reset_arena()
    bridge.close()
    print("\n[STEP 6] Engine arena reset and closed cleanly.")
    print("\n=== Demo Execution Complete ===")


if __name__ == "__main__":
    main()
