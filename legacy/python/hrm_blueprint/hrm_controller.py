from typing import List
from .configs import HRMConfig
from .blueprint import LayerLayout


class HRMControllerBlueprint:
    """
    Blueprint builder for the Hierarchical Reasoning Module (HRM).
    Constructs declarative layout for:
    - Input projection & context pooling
    - Low-level executor recurrent module (fast detailed steps)
    - High-level planner recurrent module (slow abstract steps)
    - Controller feature extraction (PEM-inspired: 9-dim features from dynamics)
    - Adaptive Computation Time (ACT) Q-head for dynamic early halting
    - Annealed restart perturbation (PEM: sigma0 * gamma^k kick on false attractors)
    """

    @staticmethod
    def build_layers(config: HRMConfig) -> List[LayerLayout]:
        layers: List[LayerLayout] = []
        d = config.hidden_dim

        # 1. State Pooling / Context Projection
        layers.append(
            LayerLayout(
                name="hrm.context_pooler",
                op_type="context_pool",
                in_dim=[-1, -1, d],
                out_dim=[-1, d],
                weight_shapes={"proj": [d, d]},
                hyperparams={"pool_method": "mean_or_first"},
            )
        )

        # 2. Low-Level Executor Module (GRU)
        layers.append(
            LayerLayout(
                name="hrm.low_level_executor",
                op_type="recurrent_executor",
                in_dim=[-1, d],
                out_dim=[-1, d],
                weight_shapes={
                    "w_low": [3 * d, d],
                    "u_low": [3 * d, d],
                    "b_low": [3 * d],
                },
                hyperparams={"substeps": config.low_level_substeps},
            )
        )

        # 3. High-Level Planner Module (GRU)
        layers.append(
            LayerLayout(
                name="hrm.high_level_planner",
                op_type="recurrent_planner",
                in_dim=[-1, d],
                out_dim=[-1, d],
                weight_shapes={
                    "w_high": [3 * d, d],
                    "u_high": [3 * d, d],
                    "b_high": [3 * d],
                },
                hyperparams={"max_steps": config.max_reasoning_steps},
            )
        )

        # 4. Controller Feature Extractor (PEM-inspired, no learned weights)
        # Computes: residual, residual_slope, energy, diversity, memory_confidence,
        #           verification_margin, plateau_count, restarts_used → 9-dim feature
        layers.append(
            LayerLayout(
                name="hrm.controller_features",
                op_type="controller_featurizer",
                in_dim=[-1, d],   # z_h at current step
                out_dim=[-1, 9],  # ControllerFeatures vector
                weight_shapes={},  # zero-param (computed from dynamics)
                hyperparams={
                    "eps_kinetic":    config.eps_kinetic,
                    "eps_energy":     config.eps_energy,
                    "tau_verify":     config.tau_verify,
                    "plateau_cycles": config.plateau_cycles,
                },
            )
        )

        # 5. ACT Q-Head: 9-feature linear Q (not full z_h projection)
        #    W_q: (2, 9), b_q: (2) — tiny, data-efficient, follows PEM design
        #    Hard rails: ACCEPT/ABSTAIN are conjunctive conditions, not Q-head outputs
        layers.append(
            LayerLayout(
                name="hrm.act_qhead",
                op_type="act_halting_head",
                in_dim=[-1, 9],   # controller features (not raw z_h)
                out_dim=[-1, 2],  # [q_halt, q_continue]
                weight_shapes={
                    "w_q": [2, 9],  # 9-feature Q (PEM pattern)
                    "b_q": [2],
                },
                hyperparams={
                    "act_threshold":  config.act_threshold,
                    "use_act":        config.use_act_halting,
                    "use_hard_rails": config.use_hard_rails,
                    "eps_kinetic":    config.eps_kinetic,
                    "eps_energy":     config.eps_energy,
                    "tau_verify":     config.tau_verify,
                },
            )
        )

        # 6. Annealed Restart Perturbation (PEM: sigma0 * gamma^k phase kick)
        #    Applied when false-attractor detected; not a learned layer
        layers.append(
            LayerLayout(
                name="hrm.restart_policy",
                op_type="annealed_restart",
                in_dim=[-1, d],
                out_dim=[-1, d],
                weight_shapes={},  # zero-param
                hyperparams={
                    "sigma0":         config.sigma0_restart,
                    "gamma":          config.gamma_restart,
                    "restart_budget": config.restart_budget,
                },
            )
        )

        return layers
