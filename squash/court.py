"""Squash court model (metres) and top-down drawing helpers.

Coordinate frame: origin at the front-left floor corner as seen from the back
wall (image-left == court-left). x runs across the court (0..6.4), y runs from
the front wall (0) to the back wall (9.75).
"""

from __future__ import annotations

import numpy as np

WIDTH = 6.40
LENGTH = 9.75
SHORT_LINE = 5.44  # distance of the short line from the front wall
BOX = 1.60  # service box side
T_POINT = (WIDTH / 2, SHORT_LINE)

# Service boxes: (x0, y0, x1, y1)
LEFT_BOX = (0.0, SHORT_LINE, BOX, SHORT_LINE + BOX)
RIGHT_BOX = (WIDTH - BOX, SHORT_LINE, WIDTH, SHORT_LINE + BOX)

# Front wall marks (height above floor) - used only for drawing/validation
TIN = 0.43
SERVICE_LINE = 1.78
OUT_LINE_FRONT = 4.57

# Six-zone grid: front / mid / back x left / right
ZONE_Y_EDGES = (0.0, 3.0, SHORT_LINE + 0.8, LENGTH)  # front <3.0, mid 3.0-6.24, back >6.24


def zone_of(x: np.ndarray, y: np.ndarray):
    """Return (row, col) zone indices: row 0=front,1=mid,2=back; col 0=left,1=right."""
    row = np.digitize(y, ZONE_Y_EDGES[1:3])
    col = (x >= WIDTH / 2).astype(int)
    return row, col


ZONE_NAMES = [["front-left", "front-right"], ["mid-left", "mid-right"], ["back-left", "back-right"]]


def in_box(x, y, box):
    x0, y0, x1, y1 = box
    return (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)


def dist_to_t(x, y):
    return np.hypot(x - T_POINT[0], y - T_POINT[1])


def court_lines():
    """Line segments (x0,y0,x1,y1) in metres for drawing a top-down court."""
    segs = [
        (0, 0, WIDTH, 0),
        (0, LENGTH, WIDTH, LENGTH),
        (0, 0, 0, LENGTH),
        (WIDTH, 0, WIDTH, LENGTH),
        (0, SHORT_LINE, WIDTH, SHORT_LINE),
        (WIDTH / 2, SHORT_LINE, WIDTH / 2, LENGTH),
    ]
    for x0, y0, x1, y1 in (LEFT_BOX, RIGHT_BOX):
        segs += [(x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)]
    return segs


def draw_court_cv(img, scale, origin=(0, 0), color=(200, 200, 200), thick=1):
    """Draw the court outline onto an OpenCV image. scale = px per metre."""
    import cv2

    ox, oy = origin
    for x0, y0, x1, y1 in court_lines():
        cv2.line(img, (int(ox + x0 * scale), int(oy + y0 * scale)), (int(ox + x1 * scale), int(oy + y1 * scale)), color, thick, cv2.LINE_AA)
    tx, ty = T_POINT
    cv2.circle(img, (int(ox + tx * scale), int(oy + ty * scale)), max(2, int(scale * 0.08)), color, -1)
    return img


def draw_court_mpl(ax, color="#888", lw=1.0):
    """Draw the court on a matplotlib axis (front wall at the top)."""
    for x0, y0, x1, y1 in court_lines():
        ax.plot([x0, x1], [y0, y1], color=color, lw=lw, solid_capstyle="round", zorder=1)
    ax.plot(*T_POINT, marker="o", ms=3, color=color, zorder=2)
    ax.set_xlim(-0.2, WIDTH + 0.2)
    ax.set_ylim(LENGTH + 0.2, -0.2)  # front wall at top
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
