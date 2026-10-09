import numpy as np

from squash import court


def test_dimensions_are_a_standard_singles_court():
    assert court.WIDTH == 6.4 and court.LENGTH == 9.75 and court.SHORT_LINE == 5.44
    assert court.T_POINT == (3.2, 5.44)
    assert court.LEFT_BOX[2] - court.LEFT_BOX[0] == court.BOX == 1.6


def test_zones_split_front_mid_back_and_left_right():
    x = np.array([1.0, 5.0, 1.0, 5.0, 1.0, 5.0])
    y = np.array([1.0, 1.0, 4.0, 4.0, 9.0, 9.0])
    row, col = court.zone_of(x, y)
    assert row.tolist() == [0, 0, 1, 1, 2, 2]
    assert col.tolist() == [0, 1, 0, 1, 0, 1]


def test_distance_to_t_and_boxes():
    assert court.dist_to_t(np.array([3.2]), np.array([5.44]))[0] == 0
    assert court.in_box(np.array([0.5]), np.array([6.0]), court.LEFT_BOX).all()
    assert not court.in_box(np.array([3.2]), np.array([6.0]), court.LEFT_BOX).any()
