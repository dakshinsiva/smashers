import numpy as np

from squash import court
from squash.calibrate import court_to_img, distort_px, fit_line_ransac, img_to_court, undistort_px


def test_lens_model_round_trips():
    d = dict(k=-0.25, cx=288.0, cy=320.0, R0=430.0)
    pts = np.array([[10.0, 20.0], [288.0, 320.0], [500.0, 600.0], [150.0, 450.0]])
    assert np.allclose(distort_px(undistort_px(pts, d), d), pts, atol=1e-6)


def test_projection_round_trips_with_and_without_lens_model():
    H = np.array([[0.02, 0.0, -2.0], [0.0, 0.03, -5.0], [0.0, 0.001, 1.0]])
    world = np.array([[0.0, 0.0], [court.WIDTH, 0.0], list(court.T_POINT), [1.0, 9.0]])
    for calib in (H, dict(H=H, distortion=dict(k=-0.2, cx=288.0, cy=320.0, R0=430.0))):
        px = court_to_img(calib, world)
        assert np.allclose(img_to_court(calib, px), world, atol=1e-4)


def test_ransac_line_ignores_outliers():
    x = np.arange(0, 100, 2.0)
    y = 0.5 * x + 10
    pts = np.c_[x, y].astype(float)
    pts[::7, 1] += 40  # gross outliers
    a, b, q = fit_line_ransac(pts, tol=1.5)
    assert abs(a - 0.5) < 0.01 and abs(b - 10) < 0.5 and q > 0.8
