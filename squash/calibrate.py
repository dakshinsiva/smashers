"""Stage 1: floor homography from the static background frame.

Measurements (all automatic, seeded by config):
  * three red lines on the front wall (out 4.57 m, service 1.78 m, tin 0.43 m)
    -> pixel scale at the front wall, floor row, right/left wall edges
  * side-wall / floor junctions from the floor colour mask -> depth vanishing point
  * red half-court line; its far end (contrast profile) is the T (3.2, 5.44)
The short line is constructed through the T parallel to the front-wall base.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from . import court

ROOT = Path(__file__).resolve().parent.parent
RNG = np.random.default_rng(0)


def load_config(path=None):
    """Load a game config (games/*.yaml). The game file names a court profile (courts/*.yaml) holding
    the camera, ROI and calibration settings; game keys override court keys. The path comes from the
    argument or the SMASHERS_CONFIG environment variable (so single stages can be run with -m)."""
    path = path or os.environ.get("SMASHERS_CONFIG")
    if not path:
        raise SystemExit("no config given: pass a game file (python run_all.py games/<game>.yaml) or set SMASHERS_CONFIG")
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    with open(path) as f:
        game = yaml.safe_load(f)
    cfg = {}
    if game.get("court"):
        court_path = Path(game["court"])
        if not court_path.is_absolute():
            court_path = ROOT / court_path
        with open(court_path) as f:
            cfg.update(yaml.safe_load(f))
        cfg["court_name"] = cfg.pop("name", court_path.stem)
    cfg.update({k: v for k, v in game.items() if k != "court"})
    for key in ("video", "output_dir"):
        if key in cfg and not Path(cfg[key]).is_absolute():
            cfg[key] = str(ROOT / cfg[key])
    cfg["config_path"] = str(path)
    return cfg


def background_median(video, t0, t1, n=60, out=None):
    cap = cv2.VideoCapture(video)
    frames = []
    for t in np.linspace(t0 + 10, t1 - 10, n):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
        if ok:
            frames.append(fr)
    bg = np.median(np.array(frames), axis=0).astype(np.uint8)
    if out:
        cv2.imwrite(str(out), bg)
    return bg


def fit_line_ransac(pts, tol=2.0, iters=800):
    """Robust y = a*x + b."""
    pts = np.asarray(pts, float)
    best, best_n = None, 0
    for _ in range(iters):
        i, j = RNG.choice(len(pts), 2, replace=False)
        (x0, y0), (x1, y1) = pts[i], pts[j]
        if abs(x1 - x0) < 1:
            continue
        a = (y1 - y0) / (x1 - x0)
        b = y0 - a * x0
        inl = np.abs(pts[:, 1] - (a * pts[:, 0] + b)) < tol
        if inl.sum() > best_n:
            best, best_n = inl, inl.sum()
    a, b = np.polyfit(pts[best, 0], pts[best, 1], 1)
    return float(a), float(b), best_n / len(pts)


def _runs(xs, gap=6, min_len=20):
    runs, s, p = [], xs[0], xs[0]
    for x in xs[1:]:
        if x > p + gap:
            runs.append((s, p))
            s = x
        p = x
    runs.append((s, p))
    return [r for r in runs if r[1] - r[0] >= min_len]


def front_wall(bg, seed):
    """Rows of the three front-wall lines, pixel scale, floor row and wall x-extent."""
    hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    red = ((H < 12) | (H > 165)) & (S > 60) & (V > 70)
    cx = int((seed["front_left"][0] + seed["front_right"][0]) / 2)
    band = red[: seed["front_left"][1], cx - 95 : cx + 95]
    cnt = band.sum(1)
    rows = np.nonzero(cnt > 0.3 * band.shape[1])[0]
    groups = []
    for y in rows:
        if groups and y - groups[-1][-1] <= 3:
            groups[-1].append(y)
        else:
            groups.append([y])
    groups = [g for g in groups if len(g) >= 1]
    assert len(groups) >= 3, f"expected 3 red lines on the front wall, found {len(groups)}"
    y_out, y_serv, y_tin = [float(np.mean(g)) for g in groups[-3:]]
    scale = (y_serv - y_out) / (court.OUT_LINE_FRONT - court.SERVICE_LINE)  # px per metre
    y_floor = y_tin + court.TIN * scale
    # x extent: right end of the service/tin lines (left end may be occluded by the door bar)
    ends = []
    for y in (y_serv, y_tin):
        rr = red[int(y) - 2 : int(y) + 3].any(0)
        xs = np.nonzero(rr)[0]
        runs = _runs(xs)
        ends.append((runs[0][0], runs[-1][1]))
    x_right = float(np.mean([e[1] for e in ends]))
    x_left_seen = float(min(e[0] for e in ends))
    x_left = x_right - court.WIDTH * scale
    return dict(
        y_out=y_out,
        y_service=y_serv,
        y_tin=y_tin,
        scale_px_per_m=float(scale),
        y_floor=float(y_floor),
        x_left=float(x_left),
        x_right=x_right,
        x_left_seen=x_left_seen,
    )


def floor_mask(bg):
    hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    m = ((S > 40) & (H >= 5) & (H <= 35) & (V > 60)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > 800]
    return np.isin(lab, keep), cv2.GaussianBlur(S.astype(np.float32), (5, 5), 0)


def side_junctions(bg, fw):
    fm, Ss = floor_mask(bg)
    h, w = fm.shape
    y0 = int(fw["y_floor"]) - 15

    def boundary(x0, x1):
        pts = []
        for x in range(x0, x1, 3):
            ys = np.nonzero(fm[y0:, x])[0]
            if not len(ys):
                continue
            yt = ys[0] + y0
            a, b = max(y0, yt - 16), min(h - 1, yt + 8)
            g = np.gradient(Ss[a:b, x])
            pts.append((x, a + int(np.argmax(g))))
        return pts

    xl, xr = int(fw["x_left"]), int(fw["x_right"])
    a_l, b_l, q_l = fit_line_ransac(boundary(int(0.25 * xl), xl - 35))
    a_r, b_r, q_r = fit_line_ransac(boundary(xr + 100, w - 12))
    return dict(left=(a_l, b_l), right=(a_r, b_r), quality=dict(left=q_l, right=q_r))


def half_court_line(bg, fw, seed):
    b, g, r = [bg[:, :, i].astype(float) for i in range(3)]
    redness = r - np.maximum(g, b)
    yf = int(fw["y_floor"])
    tx = seed["t_point"][0]
    pts = []
    for y in range(yf + 35, yf + 110):
        row = redness[y, tx - 30 : tx + 30]
        wgt = np.clip(row - row.mean(), 0, None)
        xs = np.arange(tx - 30, tx + 30)
        pts.append((y, (wgt * xs).sum() / wgt.sum()))
    pts = np.array(pts)
    a, b0 = np.polyfit(pts[:, 0], pts[:, 1], 1)  # x = a*y + b0
    contrast = []
    for y in range(yf + 8, yf + 60):
        x = int(round(a * y + b0))
        on = redness[y, x - 1 : x + 2].mean()
        off = np.r_[redness[y, x - 9 : x - 4], redness[y, x + 5 : x + 10]].mean()
        contrast.append(on - off)
    contrast = np.array(contrast)
    thr = 3.0
    y_t = None
    for i in range(len(contrast) - 2):
        if (contrast[i : i + 3] > thr).all():
            y_t = yf + 8 + i
            break
    assert y_t is not None, "could not find the far end of the half-court line"
    return dict(a=float(a), b=float(b0), t_point=(float(a * y_t + b0), float(y_t)), contrast_profile=[float(c) for c in contrast])


def detect_features(bg, seed):
    fw = front_wall(bg, seed)
    sj = side_junctions(bg, fw)
    hc = half_court_line(bg, fw, seed)
    (a_l, b_l), (a_r, b_r) = sj["left"], sj["right"]

    def X(a1, b1, a2, b2):
        x = (b2 - b1) / (a1 - a2)
        return (float(x), float(a1 * x + b1))

    y_f = fw["y_floor"]
    front_left = (fw["x_left"], y_f)
    front_right = (fw["x_right"], y_f)
    vanish = X(a_l, b_l, a_r, b_r)
    t_point = hc["t_point"]
    short_left = X(0.0, t_point[1], a_l, b_l)  # front base is level -> short line is level
    short_right = X(0.0, t_point[1], a_r, b_r)
    # junction lines should pass close to the front corners
    corner_resid = dict(left=float(a_l * front_left[0] + b_l - y_f), right=float(a_r * front_right[0] + b_r - y_f))
    # implied camera distance behind the front wall via the cross ratio on the centre line
    D = (t_point[1] - vanish[1]) / (t_point[1] - y_f) * court.SHORT_LINE
    return dict(
        front_wall=fw,
        junctions=sj,
        half_court=hc,
        front_left=front_left,
        front_right=front_right,
        t_point=t_point,
        short_left=short_left,
        short_right=short_right,
        vanish=vanish,
        corner_residual_px=corner_resid,
        implied_camera_distance_m=float(D),
        half_court_x_at_vanish_row=float(hc["a"] * vanish[1] + hc["b"]),
    )


def homography_from_features(feat):
    img = np.array([feat["front_left"], feat["front_right"], feat["t_point"], feat["short_left"], feat["short_right"]], np.float32)
    wld = np.array([(0, 0), (court.WIDTH, 0), court.T_POINT, (0, court.SHORT_LINE), (court.WIDTH, court.SHORT_LINE)], np.float32)
    H, _ = cv2.findHomography(img, wld, 0)
    err = np.linalg.norm(cv2.perspectiveTransform(img[None], H)[0] - wld, axis=1)
    return H, err


def _split_calib(calib):
    """Accept either a 3x3 homography or a calibration dict {H, distortion?}."""
    if isinstance(calib, dict):
        return np.asarray(calib["H"], np.float64), calib.get("distortion")
    return np.asarray(calib, np.float64), None


def undistort_px(pts, d):
    """Division-model radial undistortion: p_u = c + (p - c) / (1 + k r^2), r normalised by R0."""
    p = np.asarray(pts, float).reshape(-1, 2)
    c = np.array([d["cx"], d["cy"]])
    v = p - c
    r2 = (v**2).sum(1) / d["R0"] ** 2
    return c + v / (1 + d["k"] * r2)[:, None]


def distort_px(pts, d, iters=30):
    """Inverse of undistort_px (fixed-point iteration)."""
    u = np.asarray(pts, float).reshape(-1, 2)
    c = np.array([d["cx"], d["cy"]])
    v = u - c
    p = v.copy()
    for _ in range(iters):
        r2 = (p**2).sum(1) / d["R0"] ** 2
        p = v * (1 + d["k"] * r2)[:, None]
    return c + p


def img_to_court(calib, pts):
    """Image pixels -> court metres. `calib` is a 3x3 H (no lens model) or a calibration dict."""
    H, d = _split_calib(calib)
    p = np.asarray(pts, np.float32).reshape(-1, 2)
    if d:
        p = undistort_px(p, d).astype(np.float32)
    return cv2.perspectiveTransform(p.reshape(-1, 1, 2), H).reshape(-1, 2)


def court_to_img(calib, pts):
    H, d = _split_calib(calib)
    p = np.asarray(pts, np.float32).reshape(-1, 1, 2)
    u = cv2.perspectiveTransform(p, np.linalg.inv(H)).reshape(-1, 2)
    return distort_px(u, d) if d else u


def load_calibration(out_dir):
    return json.load(open(Path(out_dir) / "calibration.json"))


def overlay(bg, calib, feat, max_depth=None):
    out = bg.copy()
    if max_depth is None:
        max_depth = court.LENGTH if (isinstance(calib, dict) and calib.get("distortion")) else 9.0

    def P(x, y):
        return tuple(int(round(v)) for v in court_to_img(calib, [(x, y)])[0])

    def seg(x0, y0, x1, y1, col, n=40):
        pts = np.array([P(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t) for t in np.linspace(0, 1, n)], np.int32)
        cv2.polylines(out, [pts], False, col, 1, cv2.LINE_AA)

    for x0, y0, x1, y1 in court.court_lines():
        seg(x0, min(y0, max_depth), x1, min(y1, max_depth), (0, 255, 255))
    for y in np.arange(1, max_depth + 0.01, 1.0):
        seg(0, y, court.WIDTH, y, (255, 200, 0))
        cv2.putText(out, f"{y:.0f}m", P(court.WIDTH + 0.05, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)
    cv2.circle(out, P(*court.T_POINT), 5, (0, 0, 255), 2)
    for k, col in (
        ("front_left", (0, 255, 0)),
        ("front_right", (0, 255, 0)),
        ("t_point", (255, 0, 255)),
        ("short_left", (255, 255, 0)),
        ("short_right", (255, 255, 0)),
        ("vanish", (0, 165, 255)),
    ):
        if k not in feat:
            continue
        x, y = feat[k]
        if 0 <= x < out.shape[1] and 0 <= y < out.shape[0]:
            cv2.drawMarker(out, (int(x), int(y)), col, cv2.MARKER_CROSS, 14, 1)
    fw = feat.get("front_wall")
    if fw:
        for y in (fw["y_out"], fw["y_service"], fw["y_tin"]):
            cv2.line(out, (int(fw["x_left"]), int(y)), (int(fw["x_right"]), int(y)), (0, 0, 255), 1)
    return out


# ------------------------------------------------------------------------------------------------
# "elevated" camera profile: a CCTV-style camera mounted high and back, looking down into the court
# through strong (fisheye-ish) lens distortion. The whole floor is visible, including the back wall
# line, so corners come straight from floor-colour segmentation rather than front-wall red lines.
# ------------------------------------------------------------------------------------------------
def line_score(bg):
    """Thin dark/red painted lines on a lighter floor: black-hat + local red excess."""
    gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))).astype(float)
    b, g, r = [bg[:, :, i].astype(float) for i in range(3)]
    red = r - np.maximum(g, b)
    redbh = red - cv2.GaussianBlur(red, (21, 21), 0)
    return bh * 1.5 + np.clip(redbh, 0, None) * 4


def trace_line(score, spec, minv=20):
    """Trace one painted line. spec: {"scan": "x"|"y", "range": [a,b], "window": [lo,hi], "skip": [[a,b],...]}
    scan "x": for each x in range, the best y in window (near-horizontal line); scan "y": the reverse."""
    skip = spec.get("skip", [])
    pts = []
    for v in range(spec["range"][0], spec["range"][1], spec.get("step", 2)):
        if any(a <= v <= b for a, b in skip):
            continue
        lo, hi = spec["window"]
        seg = score[lo:hi, v] if spec["scan"] == "x" else score[v, lo:hi]
        i = int(np.argmax(seg))
        if seg[i] > minv and 0 < i < len(seg) - 1:
            pts.append((v, lo + i) if spec["scan"] == "x" else (lo + i, v))
    return np.array(pts, float)


def detect_features_elevated(bg, seed):
    """Joint fit of a floor homography and a one-parameter radial lens model from the painted floor
    lines (half-court line, short line, service-box edges) plus a few point landmarks.
    seed["lines"]: {name: {"scan", "range", "window", "skip"?, "court": ["x"|"y", value]}}
    seed["points"]: {name: {"px": [x,y], "m": [x,y], "weight": w}}"""
    from scipy.optimize import least_squares

    h, w = bg.shape[:2]
    score = line_score(bg)
    lines = {n: (trace_line(score, sp), sp["court"]) for n, sp in seed["lines"].items()}
    points = {n: (np.array(p["px"], float), np.array(p["m"], float), float(p.get("weight", 1.0))) for n, p in seed["points"].items()}
    cx, cy = w / 2, h / 2
    d0 = dict(k=0.0, cx=cx, cy=cy, R0=float(np.hypot(cx, cy)))

    def project(theta, px):
        H = np.append(theta[:8], 1).reshape(3, 3)
        d = dict(d0, k=theta[8])
        u = undistort_px(px, d)
        q = np.c_[u, np.ones(len(u))] @ H.T
        return q[:, :2] / q[:, 2:3]

    def resid(theta):
        res = []
        for px, (ax, val) in lines.values():
            c = project(theta, px)
            res.append((c[:, 0] if ax == "x" else c[:, 1]) - val)
        for px, m, wt in points.values():
            res.append((project(theta, [px])[0] - m) * wt)
        return np.concatenate(res)

    # initial homography: the hand landmarks plus every corner where a traced x-line meets a traced
    # y-line within the extent of both traces (e.g. box inner edge x short line -> (1.6, 5.44))
    def fit_line(px):
        c = px.mean(0)
        u, s_, vt = np.linalg.svd(px - c)
        d = vt[0]
        return np.array([d[1], -d[0], -(d[1] * c[0] - d[0] * c[1])])  # a x + b y + c = 0

    init = [(v[0], v[1]) for v in points.values()]
    xl = [(n, px, val) for n, (px, (ax, val)) in lines.items() if ax == "x" and len(px) >= 4]
    yl = [(n, px, val) for n, (px, (ax, val)) in lines.items() if ax == "y" and len(px) >= 4]
    for nx, pxx, xv in xl:
        for ny, pxy, yv in yl:
            p = np.cross(fit_line(pxx), fit_line(pxy))
            if abs(p[2]) < 1e-9:
                continue
            p = p[:2] / p[2]
            if np.hypot(*(pxx - p).T).min() < 25 and np.hypot(*(pxy - p).T).min() < 25:
                init.append((p, np.array([xv, yv])))
    img = np.array([a for a, _ in init], np.float32)
    wld = np.array([b for _, b in init], np.float32)
    H0, _ = cv2.findHomography(img, wld, 0)
    H0 /= H0[2, 2]
    best = None
    for k0 in (-0.3, -0.15, 0.0, 0.15):
        sol = least_squares(resid, np.r_[H0.ravel()[:8], k0], loss="soft_l1", f_scale=0.1, max_nfev=6000)
        if best is None or sol.cost < best.cost:
            best = sol
    H = np.append(best.x[:8], 1).reshape(3, 3)
    dist = dict(d0, k=float(best.x[8]))
    calib = dict(H=H, distortion=dist)
    fit = {}
    for n, (px, (ax, val)) in lines.items():
        c = img_to_court(calib, px)
        e = (c[:, 0] if ax == "x" else c[:, 1]) - val
        fit[n] = dict(n_px=int(len(px)), rms_m=float(np.sqrt((e**2).mean())), max_m=float(np.abs(e).max()))
    for n, (px, m, wt) in points.items():
        c = img_to_court(calib, [px])[0]
        fit[n] = dict(px=px.tolist(), proj_m=c.round(3).tolist(), err_m=float(np.hypot(*(c - m))))
    front_left, front_right = court_to_img(calib, [(0, 0), (court.WIDTH, 0)])
    t_point = court_to_img(calib, [court.T_POINT])[0]
    short_left, short_right = court_to_img(calib, [(0, court.SHORT_LINE), (court.WIDTH, court.SHORT_LINE)])
    return dict(
        H=H,
        distortion=dist,
        fit=fit,
        cost=float(best.cost),
        front_left=front_left.tolist(),
        front_right=front_right.tolist(),
        t_point=t_point.tolist(),
        short_left=short_left.tolist(),
        short_right=short_right.tolist(),
        junctions=dict(quality={n: round(1 - min(v.get("rms_m", 0), 1), 3) for n, v in fit.items() if "rms_m" in v}),
    )


def run(cfg=None, force=False):
    cfg = cfg or load_config()
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(exist_ok=True)
    calib_path = out_dir / "calibration.json"
    if calib_path.exists() and not force:
        print("calibration exists, skipping")
        return json.load(open(calib_path))
    bg_path = out_dir / "background.png"
    bg = cv2.imread(str(bg_path)) if bg_path.exists() else background_median(cfg["video"], cfg["t_start"], cfg["t_end"], out=bg_path)
    profile = cfg.get("camera_profile", "floor_level")
    h, w = bg.shape[:2]
    if profile == "elevated":
        feat = detect_features_elevated(bg, cfg["calibration_seed"])
        calib = dict(H=feat.pop("H"), distortion=feat.pop("distortion"))
        err = np.array(
            [v["err_m"] for v in feat["fit"].values() if "err_m" in v] + [v["rms_m"] for v in feat["fit"].values() if "rms_m" in v]
        )
    else:
        feat = detect_features(bg, cfg["calibration_seed"])
        H, err = homography_from_features(feat)
        calib = dict(H=H, distortion=None)
    bottom = img_to_court(calib, [(w / 2, h - 1), (0, h - 1), (w - 1, h - 1)])
    result = dict(
        H=np.asarray(calib["H"]).tolist(),
        distortion=calib["distortion"],
        camera_profile=profile,
        image_size=[w, h],
        features=feat,
        reproj_err_m=err.tolist(),
        frame_bottom_depth_m=float(bottom[0, 1]),
        frame_bottom_corners_court=bottom.tolist(),
    )
    json.dump(result, open(calib_path, "w"), indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else o)
    cv2.imwrite(str(out_dir / "calibration_overlay.png"), overlay(bg, calib, feat))
    show = {k: v for k, v in feat.items() if k not in ("half_court", "junctions")}
    show["junction_quality"] = feat["junctions"]["quality"]
    print(
        json.dumps(show, indent=1, default=lambda o: round(o, 3) if isinstance(o, float) else (o.tolist() if hasattr(o, "tolist") else o))
    )
    print("fit errors (m):", np.round(err, 3).tolist(), f" frame-bottom depth (m): {bottom[0, 1]:.2f}")
    return result


if __name__ == "__main__":
    run(force="--force" in sys.argv)
