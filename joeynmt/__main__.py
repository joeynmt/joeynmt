import argparse
import shutil
from pathlib import Path

import torch
import torch.multiprocessing as mp

from joeynmt.config import _check_path, load_config
from joeynmt.helpers import check_version, make_model_dir
from joeynmt.helpers_for_ddp import get_logger
from joeynmt.prediction import test, translate
from joeynmt.training import train


def main():
    ap = argparse.ArgumentParser("joeynmt")

    ap.add_argument(
        "mode",
        choices=["train", "test", "translate"],
        help="Train a model or test or translate"
    )

    ap.add_argument(
        "config_path", metavar="config-path", type=str, help="Path to YAML config file"
    )

    ap.add_argument(
        "-o", "--output-path", type=str, help="Path for saving translation output"
    )

    ap.add_argument(
        "-a",
        "--save-attention",
        action="store_true",
        help="Save attention visualizations"
    )

    ap.add_argument("-s", "--save-scores", action="store_true", help="Save scores")

    ap.add_argument(
        "-t", "--skip-test", action="store_true", help="Skip test after training"
    )

    ap.add_argument(
        "-d", "--use-ddp", action="store_true", help="Invoke DDP environment"
    )

    args = ap.parse_args()

    # read config file
    cfg = load_config(Path(args.config_path))

    # make model_dir
    if args.mode == "train":
        make_model_dir(
            Path(cfg["model_dir"]), overwrite=cfg["training"].get("overwrite", False)
        )
    model_dir = _check_path(cfg["model_dir"], allow_empty=False)
    if args.mode == "train":
        # store a copy of original training config in model dir
        # (called in __main__.py, because `args.config_path` is accessible here.)
        shutil.copy2(args.config_path, (model_dir / "config.yaml").as_posix())

    # make logger
    logger = get_logger("", log_file=Path(model_dir / f"{args.mode}.log").as_posix())
    pkg_version = check_version(cfg.get("joeynmt_version", None))
    logger.info("Hello! This is Joey-NMT (version %s).", pkg_version)
    # TODO: save version number in model checkpoints

    if args.use_ddp:
        n_gpu = torch.cuda.device_count() \
            if cfg.get("use_cuda", False) and torch.cuda.is_available() else 0
        if args.mode == "train":
            assert n_gpu > 1, "For DDP training, `world_size` must be > 1."
            logger.info("Spawn torch.multiprocessing (nprocs=%d).", n_gpu)
            cfg["use_ddp"] = args.use_ddp
            mp.spawn(train, args=(n_gpu, cfg, args.skip_test), nprocs=n_gpu)
        elif args.mode == "test":
            raise RuntimeError("For testing mode, DDP is currently not available.")
        elif args.mode == "translate":
            raise RuntimeError(
                "For interactive translation mode, "
                "DDP is currently not available."
            )

    else:
        if args.mode == "train":
            train(rank=0, world_size=None, cfg=cfg, skip_test=args.skip_test)
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
