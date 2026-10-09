"""Train the player-identity classifier from hand-labelled tracklets -> output/identity_probs.parquet

Features: keypoint-anchored colour patches (patches.parquet), HSV histograms (appearance.parquet)
and the ResNet18 embedding (cnn_features.npy). Labels: output/tracklet_labels.json, whose ids refer
to the continuity tracklets stored in output/tracklets_tmp.parquet.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict

from .calibrate import load_config


def feature_table(out_dir):
    det = (
        pd.read_parquet(out_dir / "detections.parquet").reset_index().rename(columns={"index": "det_id"})[["det_id", "frame", "t", "conf"]]
    )
    pt = pd.read_parquet(out_dir / "patches.parquet").set_index("det_id")
    app = pd.read_parquet(out_dir / "appearance.parquet").set_index("det_id")
    dcols = [c for c in app.columns if c.startswith("d")]
    X = det.join(pt, on="det_id").join(app[dcols + ["shirt_v", "shorts_s", "shorts_v"]], on="det_id")
    for c in ("h", "s", "v"):
        X[f"th_{c}"] = X[[f"lth_{c}", f"rth_{c}"]].mean(axis=1)
    hand = ["chest_s", "chest_v", "th_h", "th_s", "th_v"] + dcols + ["shirt_v", "shorts_s", "shorts_v"]
    F = np.load(out_dir / "cnn_features.npy")
    ids = np.load(out_dir / "cnn_det_ids.npy")
    cnn = pd.DataFrame(F, index=ids, columns=[f"c{i}" for i in range(F.shape[1])])
    X = X.join(cnn, on="det_id")
    return X, hand, list(cnn.columns)


def run(cfg=None, force=False):
    cfg = cfg or load_config()
    out_dir = Path(cfg["output_dir"])
    out = out_dir / "identity_probs.parquet"
    if out.exists() and not force:
        return pd.read_parquet(out)
    X, hand, cnn = feature_table(out_dir)
    lab = json.load(open(out_dir / "tracklet_labels.json"))
    tl = pd.read_parquet(out_dir / "tracklets_tmp.parquet")[["det_id", "tracklet"]]
    X = X.merge(tl, on="det_id", how="left")
    # classes: 0 = p2, 1 = p1, 2 = other (spectators / non-players), -1 = unlabelled
    X["label"] = np.where(
        X.tracklet.isin(lab["p1"]), 1, np.where(X.tracklet.isin(lab["p2"]), 0, np.where(X.tracklet.isin(lab.get("other", [])), 2, -1))
    )
    feats = hand + cnn
    Xf = X[feats].copy()
    Xf[hand] = Xf[hand].fillna(Xf[hand].median())
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-9
    Z = ((Xf - mu) / sd).to_numpy()
    L = X.label.to_numpy() >= 0
    y, g = X.label.to_numpy()[L], X.tracklet.to_numpy()[L]
    clf = LogisticRegression(C=0.3, max_iter=5000)
    classes = sorted(set(y.tolist()))
    proba = cross_val_predict(clf, Z[L], y, groups=g, cv=GroupKFold(5), method="predict_proba")
    pred = np.array(classes)[proba.argmax(1)]
    tr = (
        pd.DataFrame(dict(g=g, pred=pred, y=y))
        .groupby("g")
        .agg(pred=("pred", lambda v: v.value_counts().index[0]), y=("y", "first"), n=("pred", "size"))
    )
    names = {0: "p2", 1: "p1", 2: "other"}
    print(
        f"identity model ({len(classes)} classes {[names[c] for c in classes]}): {L.sum()} labelled detections in {len(tr)} tracklets; "
        f"group-CV accuracy per detection {(pred == y).mean():.3f}, per tracklet {(tr.pred == tr.y).mean():.3f}"
    )
    bad = tr[tr.pred != tr.y]
    if len(bad):
        print(
            "  tracklets disagreeing with their label:",
            {int(k): dict(pred=names[int(v.pred)], label=names[int(v.y)], n=int(v.n)) for k, v in bad.iterrows()},
        )
    clf.fit(Z[L], y)
    P = clf.predict_proba(Z)
    col = {c: i for i, c in enumerate(clf.classes_)}
    X["prob_p2"] = P[:, col[0]] if 0 in col else 0.0
    X["prob_p1"] = P[:, col[1]] if 1 in col else 0.0
    X["prob_other"] = P[:, col[2]] if 2 in col else 0.0
    X[["det_id", "prob_p1", "prob_p2", "prob_other"]].to_parquet(out, index=False)
    print(
        "prob_p1 histogram:",
        np.histogram(X.prob_p1, bins=np.linspace(0, 1, 11))[0].tolist(),
        " prob_other>0.5:",
        int((X.prob_other > 0.5).sum()),
    )
    return X[["det_id", "prob_p1", "prob_p2", "prob_other"]]


if __name__ == "__main__":
    run(force=True)
