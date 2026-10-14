"""Where every neuron of the simulated brain sits, for the brain viewer.

Positions are the FlyWire cell-body (soma) coordinates from the annotation
table (brain/data/flywire_annotations.tsv.gz), falling back to the neuron's
representative point when no soma was annotated. They are converted to
micrometres and centred, in the brain's own frame:
  x  the fly's left (-) to right (+)
  y  dorsal (+) to ventral (-)
  z  anterior (+) to posterior (-)

Also picks the "wiring" the viewer draws: for each neuron its single
strongest outgoing connection, keeping the strongest MAX_EDGES of those, so
the picture shows real connectome edges without drawing all 15M synapses.

The layout is index-aligned with the brain engine (FastBrain / NpBrain
index = position in Completeness_783.csv), and cached to
brain/data/viz_layout.npz so replays don't need the brain.
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "brain" / "data"
ANNOTATIONS = DATA / "flywire_annotations.tsv.gz"
CACHE = DATA / "viz_layout.npz"
MAX_EDGES = 25000

# super_class -> small int, for the viewer's colours
CLASSES = ["optic", "central", "sensory", "visual_projection", "ascending", "descending",
           "sensory_ascending", "visual_centrifugal", "motor", "endocrine", "unknown"]


def _annotations():
    import pandas as pd
    cols = ["root_id", "pos_x", "pos_y", "pos_z", "soma_x", "soma_y", "soma_z", "super_class"]
    return pd.read_csv(ANNOTATIONS, sep="\t", usecols=cols, low_memory=False).set_index("root_id")


def build(ids, indptr=None, post_idx=None, post_w=None):
    """ids: FlyWire root ids in engine index order. The CSR arrays (from the
    engine: brain.indptr / post_idx / post_w) are optional; without them no
    edges are drawn."""
    ids = np.asarray(ids, dtype=np.int64)
    ann = _annotations().reindex(ids)
    xyz = np.column_stack([ann[f"soma_{a}"].fillna(ann[f"pos_{a}"]) for a in "xyz"]).astype(np.float64)
    # FlyWire voxels are 4 x 4 x 40 nm
    xyz *= np.array([0.004, 0.004, 0.040])
    missing = np.isnan(xyz).any(axis=1)
    centre = np.nanmedian(xyz, axis=0)
    xyz = xyz - centre
    xyz[:, 1] *= -1          # FlyWire y grows ventrally; viewer y grows up
    xyz[:, 2] *= -1          # FlyWire z grows posteriorly; viewer z grows toward the face
    xyz[missing] = 0.0
    cls = ann["super_class"].fillna("unknown").map({c: i for i, c in enumerate(CLASSES)}).fillna(
        len(CLASSES) - 1).to_numpy(np.uint8)

    edges = np.zeros((0, 2), np.int32)
    ew = np.zeros(0, np.float32)
    if indptr is not None:
        indptr = np.asarray(indptr)
        n = len(ids)
        has = np.flatnonzero(np.diff(indptr) > 0)
        aw = np.abs(np.asarray(post_w))
        # strongest outgoing synapse per neuron
        best = np.maximum.reduceat(aw, indptr[has])
        pos = np.empty(len(has), np.int64)
        for k, i in enumerate(has):
            s, e = indptr[i], indptr[i + 1]
            pos[k] = s + int(np.argmax(aw[s:e]))
        keep = np.argsort(-best)[:MAX_EDGES]
        pre = has[keep]
        post = np.asarray(post_idx)[pos[keep]]
        ok = ~missing[pre] & ~missing[post] & (pre != post) & (post < n)
        edges = np.column_stack([pre[ok], post[ok]]).astype(np.int32)
        ew = np.asarray(post_w)[pos[keep]][ok].astype(np.float32)
    return {"ids": ids, "xyz": xyz.astype(np.float32), "cls": cls, "missing": missing,
            "edges": edges, "edge_w": ew}


def from_brain(brain, cache=True):
    """Layout for a live engine (FastBrain or NpBrain)."""
    n = max(brain.flyid2idx.values()) + 1
    ids = np.zeros(n, np.int64)
    for f, i in brain.flyid2idx.items():
        ids[i] = f
    lay = build(ids, brain.indptr, brain.post_idx, brain.post_w)
    if cache:
        save(lay)
    return lay


def save(lay, path=CACHE):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **lay)


def load(path=CACHE):
    """The cached layout; built from the connectome export if there's no cache."""
    if Path(path).exists():
        d = np.load(path)
        return {k: d[k] for k in d.files}
    npz = DATA / "connectome_783.npz"
    if not npz.exists():
        sys.exit(f"no {path.name} and no {npz.name}: run the fly once with --brain-viz, "
                 "or python brain/export_connectome_npz.py")
    d = np.load(npz)
    pre, post, w = d["pre"].astype(np.int64), d["post"].astype(np.int64), d["w"].astype(np.float32)
    order = np.argsort(pre, kind="stable")
    indptr = np.searchsorted(pre[order], np.arange(len(d["ids"]) + 1))
    lay = build(d["ids"], indptr, post[order], w[order])
    save(lay, path)
    return lay


def groups(index_of):
    """Named neuron groups the viewer labels: the fly's eyes (the visual
    populations the senses drive) and its descending neurons (the readout).
    index_of: FlyWire id -> engine index."""
    sys.path.insert(0, str(ROOT / "brain"))
    import neurons
    scene = json.loads((ROOT / "brain" / "scene_neurons.json").read_text())
    pops = {("LC4", "left"): neurons.LC4_LOOMING_LEFT, ("LC4", "right"): neurons.LC4_LOOMING_RIGHT,
            ("LPLC4", "left"): neurons.LPLC4_LOOMING_LEFT, ("LPLC4", "right"): neurons.LPLC4_LOOMING_RIGHT}
    for t in ("LC6", "LC11", "LC12", "LPLC2", "LC22", "VS", "HS"):
        pops[(t, "left")], pops[(t, "right")] = scene[t]["left"], scene[t]["right"]
    out = {f"{t}_{s}": [index_of[int(f)] for f in fids if int(f) in index_of] for (t, s), fids in pops.items()}
    out["readout_dns"] = [index_of[int(d["id"])] for d in scene["readout_dns_v3"] if int(d["id"]) in index_of]
    out["all_dns"] = [index_of[int(f)] for f in scene["all_dns"] if int(f) in index_of]
    return out


def to_json(lay):
    """Compact JSON for the viewer: base64 typed arrays + the group indices."""
    import base64
    b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()  # noqa: E731
    index_of = {int(f): i for i, f in enumerate(lay["ids"])}
    return json.dumps({"n": int(len(lay["ids"])), "classes": CLASSES,
                       "xyz": b64(lay["xyz"].astype(np.float32)), "cls": b64(lay["cls"].astype(np.uint8)),
                       "edges": b64(lay["edges"].astype(np.uint32)), "edge_w": b64(lay["edge_w"].astype(np.float32)),
                       "groups": groups(index_of)})


if __name__ == "__main__":
    lay = load()
    print(f"{len(lay['ids'])} neurons ({int(lay['missing'].sum())} without a position), "
          f"{len(lay['edges'])} edges -> {CACHE}")
    gz = gzip.compress(to_json(lay).encode())
    print(f"viewer payload {len(gz) / 1e6:.1f} MB gzipped")
