import argparse
import shutil
from pathlib import Path

from joeynmt.config import _check_path, load_config
from joeynmt.helpers import check_version, make_logger, make_model_dir
from joeynmt.prediction import test, translate
from joeynmt.training import train


def main():
    ap = argparse.ArgumentParser("Joey NMT")

    ap.add_argument("mode",
                    choices=["train", "test", "translate"],
                    help="Train a model or test or translate")

    ap.add_argument("config_path",
                    metavar="config-path",
                    type=str,
                    help="Path to YAML config file")

    ap.add_argument("-o",
                    "--output-path",
                    type=str,
                    help="Path for saving translation output")

    ap.add_argument("-a",
                    "--save-attention",
                    action="store_true",
                    help="Save attention visualizations")

    ap.add_argument("-s", "--save-scores", action="store_true", help="Save scores")

    ap.add_argument("-t",
                    "--skip-test",
                    action="store_true",
                    help="Skip test after training")

    args = ap.parse_args()

    # read config file
    cfg = load_config(Path(args.config_path))

    # make model_dir
    if args.mode == "train":
        make_model_dir(Path(cfg["model_dir"]),
                       overwrite=cfg["training"].get("overwrite", False))
    model_dir = _check_path(cfg["model_dir"], allow_empty=False)

    # make logger
    pkg_version = make_logger(model_dir, mode=args.mode)
    # TODO: save version number in model checkpoints
    if "joeynmt_version" in cfg:
        check_version(pkg_version, cfg["joeynmt_version"])

    if args.mode == "train":
        # store copy of original training config in model dir
        shutil.copy2(args.config_path, (model_dir / "config.yaml").as_posix())

        train(cfg=cfg, skip_test=args.skip_test)
    elif args.mode == "test":
        test(
            cfg=cfg,
            output_path=args.output_path,
            save_attention=args.save_attention,
            save_scores=args.save_scores,
        )
    elif args.mode == "translate":
        translate(cfg=cfg, output_path=args.output_path)
    else:
        raise ValueError("Unknown mode")


if __name__ == "__main__":
    main()
