import argparse

from joeynmt.prediction import test, translate
from joeynmt.training import train


def main():
    ap = argparse.ArgumentParser("Joey NMT")

    ap.add_argument(
        "mode",
        choices=["train", "test", "translate"],
        help="train a model or test or translate",
    )

    ap.add_argument("config_path", type=str, help="path to YAML config file")

    ap.add_argument("-c", "--ckpt", type=str, help="checkpoint for prediction")

    ap.add_argument("-o",
                    "--output_path",
                    type=str,
                    help="path for saving translation output")

    ap.add_argument(
        "-a",
        "--save_attention",
        action="store_true",
        help="save attention visualizations",
    )

    ap.add_argument("-s", "--save_scores", action="store_true", help="save scores")

    ap.add_argument(
        "-t",
        "--skip_test",
        action="store_true",
        help="Skip test after training",
    )

    args = ap.parse_args()

    if args.mode == "train":
        train(cfg_file=args.config_path, skip_test=args.skip_test)
    elif args.mode == "test":
        test(
            cfg_file=args.config_path,
            ckpt=args.ckpt,
            output_path=args.output_path,
            save_attention=args.save_attention,
            save_scores=args.save_scores,
        )
    elif args.mode == "translate":
        translate(
            cfg_file=args.config_path,
            ckpt=args.ckpt,
            output_path=args.output_path,
        )
    else:
        raise ValueError("Unknown mode")


if __name__ == "__main__":
    main()
