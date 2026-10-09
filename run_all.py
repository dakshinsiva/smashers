"""Run the whole analysis pipeline for one game. Stages with existing outputs are skipped unless
--force (all) or --force-from <stage> is given.

  python run_all.py games/deepa_vs_arav.yaml
  python run_all.py games/deepa_vs_arav.yaml --force-from identify
  python run_all.py games/deepa_vs_arav.yaml --no-video      # skip the annotated video
"""

import os
import sys
from pathlib import Path

from squash import appearance, calibrate, detect, figures, identify, identity_model, metrics, patches, rallies, render, report, strengths

STAGES = [
    "calibrate",
    "detect",
    "appearance",
    "identity_model",
    "identify",
    "rallies",
    "metrics",
    "figures",
    "report",
    "strengths",
    "render",
]


def main(argv):
    paths = [a for a in argv if a.endswith((".yaml", ".yml"))]
    if paths:
        os.environ["SMASHERS_CONFIG"] = paths[0]
    cfg = calibrate.load_config(paths[0] if paths else None)
    print(f"game config: {cfg['config_path']}  court: {cfg.get('court_name', '?')}  output: {cfg['output_dir']}")
    force_all = "--force" in argv
    force_from = argv[argv.index("--force-from") + 1] if "--force-from" in argv else None
    start = STAGES.index(force_from) if force_from else len(STAGES)

    def f(stage):
        return force_all or STAGES.index(stage) >= start

    calibrate.run(cfg, force=f("calibrate"))
    detect.run(cfg, force=f("detect"))
    appearance.run(cfg, force=f("appearance"))
    appearance.cnn_scores(cfg, force=f("appearance"))
    patches.run(cfg, force=f("appearance"))
    identity_model.run(cfg, force=f("identity_model"))
    identify.run(cfg, force=f("identify"))
    rallies.run(cfg, force=f("rallies"))
    metrics.run(cfg, force=True)
    figures.run(cfg, force=True)
    report.run(cfg)
    report.to_pdf(cfg)
    strengths.run(cfg)
    strengths.to_pdf(cfg)
    if "--no-video" not in argv and (f("render") or not Path(cfg["output_dir"], "annotated.mp4").exists()):
        render.run(cfg, rallies_only="--rallies-only" in argv)


if __name__ == "__main__":
    main(sys.argv[1:])
