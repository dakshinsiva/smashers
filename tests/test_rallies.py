from squash.rallies import reconstruct_score, serve_clusters


def rally(server, box):
    return dict(server=server, box=box)


def test_par_scoring_winner_serves_next():
    rallies = [rally("p1", "right"), rally("p1", "left"), rally("p2", "right"), rally("p1", "left")]
    out = reconstruct_score(rallies)
    assert [o["winner"] for o in out] == ["p1", "p2", "p1", None]
    assert (out[-1]["score_p1"], out[-1]["score_p2"]) == (2, 1)
    assert "unknown" in out[-1]["note"]


def test_same_server_same_box_is_a_let_not_a_point():
    out = reconstruct_score([rally("p1", "right"), rally("p1", "right"), rally("p2", "left")])
    assert out[0]["winner"] is None and out[0]["note"].startswith("let")
    assert out[1]["winner"] == "p2"


def test_game_ball_decides_the_last_rally():
    # p1 serves fifteen rallies in a row, alternating boxes: fourteen won points, then the game ball
    seq = [rally("p1", "right" if i % 2 == 0 else "left") for i in range(15)]
    out = reconstruct_score(seq)
    assert [o["winner"] for o in out[:-1]] == ["p1"] * 14
    assert out[-1]["winner"] == "p1" and (out[-1]["score_p1"], out[-1]["score_p2"]) == (15, 0)


def test_serve_clusters_merge_flicker_and_take_majority():
    events = [
        dict(server="p1", box="left", t0=10.0, t1=10.4),
        dict(server="p2", box="right", t0=10.6, t1=10.8),
        dict(server="p1", box="left", t0=11.0, t1=12.5),
        dict(server="p2", box="right", t0=30.0, t1=30.6),
    ]
    clusters = serve_clusters(events, gap=3.0)
    assert len(clusters) == 2
    assert (clusters[0]["server"], clusters[0]["box"]) == ("p1", "left")
    assert clusters[1]["server"] == "p2"
