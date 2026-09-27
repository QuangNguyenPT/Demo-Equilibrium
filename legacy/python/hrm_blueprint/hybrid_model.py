from .configs import ModelConfig
from .blueprint import ModelBlueprint, LayerLayout
from .transformer import TransformerBackboneBlueprint
from .hrm_controller import HRMControllerBlueprint


class HybridHRMLLMBlueprint:
    """
    Unified Architecture Blueprint uniting:
    1. Language Model Backbone (Transformer)
    2. Hierarchical Reasoning Engine (HRM with ACT)
    3. Output Decoder Head (Next-token / Solution Logits)
    """

    @classmethod
    def create(cls, config: ModelConfig) -> ModelBlueprint:
        blueprint = ModelBlueprint(name=config.model_name)

        # 1. Add Backbone Layers
        for layer in TransformerBackboneBlueprint.build_layers(config.vocab_size, config.transformer):
            blueprint.add_layer(layer)

        # 2. Add HRM Reasoning Modules
        for layer in HRMControllerBlueprint.build_layers(config.hrm):
            blueprint.add_layer(layer)

        # 3. Add Language / Output Decoder Head
        d = config.transformer.hidden_dim
        blueprint.add_layer(
            LayerLayout(
                name="lm_decoder_head",
                op_type="linear_head",
                in_dim=[-1, d],
                out_dim=[-1, config.vocab_size],
                weight_shapes={"weight": [config.vocab_size, d]},
                hyperparams={"vocab_size": config.vocab_size},
            )
        )

        return blueprint

    @classmethod
    def modify_reasoning_depth(cls, blueprint: ModelBlueprint, new_max_steps: int) -> None:
        """Example of rapid Python blueprint modification without touching C++."""
        for layer in blueprint.layers:
            if layer.name == "hrm.high_level_planner":
                layer.hyperparams["max_steps"] = new_max_steps
                print(f"[BLUEPRINT] Updated reasoning max_steps -> {new_max_steps}")
                return
