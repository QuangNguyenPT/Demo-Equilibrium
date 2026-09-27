from typing import List
from .configs import TransformerConfig
from .blueprint import LayerLayout


class TransformerBackboneBlueprint:
    """
    Blueprint builder for Transformer LLM backbone.
    Constructs declarative layer representations for:
    - Embedding
    - RMSNorm layers
    - Multi-Head Attention (with Q, K, V projections and FlashAttention layout)
    - SwiGLU MLP (Gate, Up, Down projections)
    """

    @staticmethod
    def build_layers(vocab_size: int, config: TransformerConfig) -> List[LayerLayout]:
        layers: List[LayerLayout] = []
        d = config.hidden_dim
        inter_d = config.intermediate_dim

        # 1. Token Embedding
        layers.append(
            LayerLayout(
                name="token_embeddings",
                op_type="embedding",
                in_dim=[-1, -1],  # (B, T)
                out_dim=[-1, -1, d],  # (B, T, D)
                weight_shapes={"weight": [vocab_size, d]},
                hyperparams={"vocab_size": vocab_size, "hidden_dim": d},
            )
        )

        # 2. Transformer Blocks
        for i in range(config.num_layers):
            prefix = f"layer_{i}"

            # Pre-attention RMSNorm
            layers.append(
                LayerLayout(
                    name=f"{prefix}.attn_norm",
                    op_type="rms_norm",
                    in_dim=[-1, -1, d],
                    out_dim=[-1, -1, d],
                    weight_shapes={"gamma": [d]},
                    hyperparams={"eps": config.norm_eps},
                )
            )

            # Self-Attention Block
            layers.append(
                LayerLayout(
                    name=f"{prefix}.attention",
                    op_type="flash_attention",
                    in_dim=[-1, -1, d],
                    out_dim=[-1, -1, d],
                    weight_shapes={
                        "wq": [d, d],
                        "wk": [d, d],
                        "wv": [d, d],
                        "wo": [d, d],
                    },
                    hyperparams={
                        "num_heads": config.num_attention_heads,
                        "head_dim": config.head_dim,
                        "rope_base": config.rope_base,
                        "is_causal": True,
                        "use_attn_residuals": config.use_attn_residuals,
                        "attn_residual_block_size": config.attn_residual_block_size,
                        "layer_idx": i,
                    },
                )
            )

            # Pre-FFN RMSNorm
            layers.append(
                LayerLayout(
                    name=f"{prefix}.ffn_norm",
                    op_type="rms_norm",
                    in_dim=[-1, -1, d],
                    out_dim=[-1, -1, d],
                    weight_shapes={"gamma": [d]},
                    hyperparams={"eps": config.norm_eps},
                )
            )

            # SwiGLU Feed-Forward Network
            layers.append(
                LayerLayout(
                    name=f"{prefix}.swiglu_mlp",
                    op_type="swiglu_mlp",
                    in_dim=[-1, -1, d],
                    out_dim=[-1, -1, d],
                    weight_shapes={
                        "w_gate": [inter_d, d],
                        "w_up": [inter_d, d],
                        "w_down": [d, inter_d],
                    },
                    hyperparams={"intermediate_dim": inter_d},
                )
            )

        # Final Backbone Norm
        layers.append(
            LayerLayout(
                name="final_norm",
                op_type="rms_norm",
                in_dim=[-1, -1, d],
                out_dim=[-1, -1, d],
                weight_shapes={"gamma": [d]},
                hyperparams={"eps": config.norm_eps},
            )
        )

        return layers
