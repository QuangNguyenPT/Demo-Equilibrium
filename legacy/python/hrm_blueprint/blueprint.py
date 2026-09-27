from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
import json


@dataclass
class LayerLayout:
    """Declarative blueprint description of a single layer or operator."""
    name: str
    op_type: str  # 'embedding', 'rms_norm', 'attention', 'swiglu', 'hrm_reasoner', 'linear_head'
    in_dim: List[int]
    out_dim: List[int]
    weight_shapes: Dict[str, List[int]] = field(default_factory=dict)
    hyperparams: Dict[str, Any] = field(default_factory=dict)

    @property
    def parameter_count(self) -> int:
        count = 0
        for shape in self.weight_shapes.values():
            prod = 1
            for d in shape:
                prod *= d
            count += prod
        return count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "op_type": self.op_type,
            "in_dim": self.in_dim,
            "out_dim": self.out_dim,
            "weight_shapes": self.weight_shapes,
            "hyperparams": self.hyperparams,
            "param_count": self.parameter_count,
        }


class ModelBlueprint:
    """
    High-level declarative blueprint of the entire model layout.
    Allows easy addition, removal, inspection, and modification of layers
    in Python before dispatching computation to the C++ engine.
    """
    def __init__(self, name: str = "HRMModelBlueprint"):
        self.name = name
        self.layers: List[LayerLayout] = []

    def add_layer(self, layer: LayerLayout) -> None:
        self.layers.append(layer)

    @property
    def total_parameters(self) -> int:
        return sum(layer.parameter_count for layer in self.layers)

    def summary(self) -> str:
        lines = [
            f"=======================================================================",
            f" Model Blueprint Layout: {self.name}",
            f" Total Layers: {len(self.layers)} | Total Parameters: {self.total_parameters:,}",
            f"=======================================================================",
            f"{'Layer Name':<28} {'Op Type':<16} {'Out Shape':<14} {'Params':<12}",
            f"-----------------------------------------------------------------------",
        ]
        for l in self.layers:
            out_str = "x".join(str(x) for x in l.out_dim)
            lines.append(f"{l.name:<28} {l.op_type:<16} {out_str:<14} {l.parameter_count:>11,}")
        lines.append(f"=======================================================================")
        return "\n".join(lines)

    def export_graph_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.name,
            "total_params": self.total_parameters,
            "layers": [layer.to_dict() for layer in self.layers],
        }

    def export_json(self, filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.export_graph_dict(), f, indent=2)
