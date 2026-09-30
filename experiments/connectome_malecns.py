"""B1 — MaleCNS v1.0: re-measure the delivery's connectome principles and add one new one.

Local data only (C:/Users/Admin/Downloads/flybrain, Janelia male CNS v1.0, minconf 0.5):
connectome-weights (body_pre, body_post, weight), body-annotations (superclass, class, type,
status). The synapse-level tables are not needed for any question asked here.

  M1  whole-CNS reciprocity on the proofread core, against the directed configuration-model
      expectation  E[r] = (sum_i k_out_i k_in_i)^2 / m^3  (analytic, no sampling)
  M2  mushroom body: Kenyon cells, olfactory-PN claws per KC, APL -> KC coverage
      (re-measures doc 14 C2 from the raw table rather than quoting it)
  M3  NEW: one expanded code, many readouts. How many MBON types read each KC, how much the
      KC sets of different MBON types overlap, and how many dopaminergic (DAN) types gate each
      MBON type. This is the question doc 14 left open -- how a fly adds a *new* learned
      association: by a new readout of an existing sparse code, or by new representation?

Needs pyarrow + pandas (the repository's own interpreter has neither; any Python that has them
works). Run: `<python-with-pyarrow> -m experiments.connectome_malecns [--data DIR]`
(writes `experiments/results/connectome_malecns.json`).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pyarrow.compute as pc
import pyarrow.feather as pf
import pyarrow.ipc as ipc

DATA = Path(r"C:\Users\Admin\Downloads\flybrain")
RESULTS = Path(__file__).parent / "results" / "connectome_malecns.json"
EXCLUDED_STATUS = {"Orphan", "Glia", "Unimportant"}
CLAW_MIN = 3        # synapses for a PN->KC connection to count as a claw (doc 14, Zheng et al. 2022)
READ_MIN = 3        # synapses for KC->MBON and DAN->MBON connections


def load_annotations(data: Path):
    cols = ["bodyId", "superclass", "class", "type", "status"]
    a = pf.read_table(data / "body-annotations-male-cns-v1.0-minconf-0.5.feather", columns=cols).to_pandas()
    core = a[a.superclass.notna() & ~a.status.isin(EXCLUDED_STATUS)]
    return a, core


def load_edges(data: Path, bodies: np.ndarray):
    """(pre, post, weight) restricted to edges with both ends in `bodies`, streamed batch by batch."""
    keep = pc.field("body_pre").isin(bodies) & pc.field("body_post").isin(bodies)
    reader = ipc.open_file(data / "connectome-weights-male-cns-v1.0-minconf-0.5.feather")
    import pyarrow.dataset as ds
    parts = []
    for i in range(reader.num_record_batches):
        batch = ds.dataset(reader.get_batch(i)).to_table(filter=keep)
        parts.append((batch["body_pre"].to_numpy(), batch["body_post"].to_numpy(), batch["weight"].to_numpy()))
    return tuple(np.concatenate([p[k] for p in parts]) for k in range(3))


def reciprocity(pre: np.ndarray, post: np.ndarray) -> Dict[str, float]:
    """Fraction of directed edges whose reverse exists, and the configuration-model expectation."""
    ids, inv = np.unique(np.concatenate([pre, post]), return_inverse=True)
    n = len(ids)
    a, b = inv[:len(pre)].astype(np.int64), inv[len(pre):].astype(np.int64)
    fwd = a * n + b
    rev = b * n + a
    observed = float(np.isin(rev, fwd, assume_unique=False).mean())
    k_out = np.bincount(a, minlength=n).astype(np.float64)
    k_in = np.bincount(b, minlength=n).astype(np.float64)
    m = float(len(a))
    expected = float((k_out * k_in).sum() ** 2 / m ** 3)
    return {"edges": int(m), "neurons": int(n), "reciprocity": observed, "null_configuration": expected,
            "ratio": observed / expected}


def mushroom_body(ann, pre, post, w) -> Dict:
    t = ann.set_index("bodyId")
    kcs = ann.bodyId[ann["class"] == "Kenyon_Cell"].to_numpy()
    pns = ann.bodyId[ann["class"] == "ALPN"].to_numpy()
    apl = ann.bodyId[ann.type == "APL"].to_numpy()
    mbons = ann[ann["class"] == "MBON"]
    dans = ann[ann["class"] == "DAN"]
    out: Dict = {"kenyon_cells": int(len(kcs)), "alpn": int(len(pns)), "apl": int(len(apl)),
                 "mbon_bodies": int(len(mbons)), "mbon_types": int(mbons.type.nunique()),
                 "dan_bodies": int(len(dans)), "dan_types": int(dans.type.nunique())}

    # M2 claws: distinct ALPN partners (>= CLAW_MIN synapses) per KC, and distinct glomeruli.
    sel = np.isin(pre, pns) & np.isin(post, kcs) & (w >= CLAW_MIN)
    claws: Dict[int, set] = {}
    gloms: Dict[int, set] = {}
    for p, q in zip(pre[sel], post[sel]):
        claws.setdefault(int(q), set()).add(int(p))
        gloms.setdefault(int(q), set()).add(str(t.at[int(p), "type"]).split("_")[0])
    per_kc = np.array([len(claws.get(int(k), ())) for k in kcs])
    per_kc_glom = np.array([len(gloms.get(int(k), ())) for k in kcs])
    out["claws_per_kc"] = {"median": float(np.median(per_kc)), "mean": float(per_kc.mean()),
                           "kcs_without_pn_input": int((per_kc == 0).sum())}
    out["glomeruli_per_kc"] = {"median": float(np.median(per_kc_glom)), "mean": float(per_kc_glom.mean())}
    apl_kc = np.isin(pre, apl) & np.isin(post, kcs)
    out["apl_to_kc"] = {"synapses": int(w[apl_kc].sum()), "kcs_reached": int(len(np.unique(post[apl_kc])))}

    # M3 readouts: KC -> MBON type fan-out, overlap of KC sets between MBON types, DAN gating.
    mbon_type = dict(zip(mbons.bodyId, mbons.type))
    sel = np.isin(pre, kcs) & np.isin(post, mbons.bodyId.to_numpy()) & (w >= READ_MIN)
    readers: Dict[int, set] = {}
    kc_sets: Dict[str, set] = {}
    for p, q in zip(pre[sel], post[sel]):
        readers.setdefault(int(p), set()).add(mbon_type[int(q)])
        kc_sets.setdefault(mbon_type[int(q)], set()).add(int(p))
    fan = np.array([len(readers.get(int(k), ())) for k in kcs])
    out["mbon_types_per_kc"] = {"median": float(np.median(fan)), "mean": float(fan.mean()),
                                "frac_kc_read_by_ge2_types": float((fan >= 2).mean()),
                                "frac_kc_read_by_ge5_types": float((fan >= 5).mean()),
                                "frac_kc_unread": float((fan == 0).mean())}
    names = sorted(kc_sets)
    jac = [len(kc_sets[x] & kc_sets[y]) / len(kc_sets[x] | kc_sets[y])
           for i, x in enumerate(names) for y in names[i + 1:]]
    sizes = np.array([len(kc_sets[x]) for x in names])
    out["kc_sets_per_mbon_type"] = {"types_with_kc_input": len(names), "median_kcs": float(np.median(sizes)),
                                    "mean_pairwise_jaccard": float(np.mean(jac)) if jac else 0.0}
    dan_type = dict(zip(dans.bodyId, dans.type))
    sel = np.isin(pre, dans.bodyId.to_numpy()) & np.isin(post, mbons.bodyId.to_numpy()) & (w >= READ_MIN)
    gates: Dict[str, set] = {}
    for p, q in zip(pre[sel], post[sel]):
        gates.setdefault(mbon_type[int(q)], set()).add(dan_type[int(p)])
    g = np.array([len(gates.get(x, ())) for x in names])
    out["dan_types_per_mbon_type"] = {"median": float(np.median(g)), "mean": float(g.mean()),
                                      "frac_gated": float((g > 0).mean())}
    # Share of all KC output synapses that go to MBONs vs back to KCs / APL (where the code is read).
    kc_out = np.isin(pre, kcs)
    to = {"mbon": np.isin(post, mbons.bodyId.to_numpy()), "kc": np.isin(post, kcs), "apl": np.isin(post, apl),
          "dan": np.isin(post, dans.bodyId.to_numpy())}
    total = float(w[kc_out].sum())
    out["kc_output_share"] = {k: float(w[kc_out & v].sum()) / total for k, v in to.items()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA))
    args = ap.parse_args()
    data = Path(args.data)
    start = time.time()
    ann, core = load_annotations(data)
    print(f"[CNS] {len(ann)} annotated bodies, {len(core)} in the proofread core", flush=True)
    pre, post, w = load_edges(data, core.bodyId.to_numpy())
    print(f"[CNS] core edges {len(pre)}, synapses {int(w.sum())} ({time.time() - start:.0f}s)", flush=True)
    result: Dict = {"source": "Janelia male CNS v1.0, minconf 0.5 (local feather files)",
                    "core_neurons": int(len(core)), "core_edges": int(len(pre)), "core_synapses": int(w.sum())}
    result["M1_reciprocity"] = {">=1": reciprocity(pre, post), ">=5": reciprocity(pre[w >= 5], post[w >= 5])}
    print(f"[CNS] M1 {result['M1_reciprocity']}", flush=True)
    result["M2_M3_mushroom_body"] = mushroom_body(ann, pre, post, w)
    print(f"[CNS] M2/M3 {json.dumps(result['M2_M3_mushroom_body'])}", flush=True)
    result["seconds"] = round(time.time() - start)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[CNS] wrote {RESULTS}")


if __name__ == "__main__":
    main()
