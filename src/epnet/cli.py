from __future__ import annotations

import argparse
import sys

from . import ablate, evaluate, export, infer, profile, train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified EPNet CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("train")
    subparsers.add_parser("eval")
    subparsers.add_parser("infer")
    subparsers.add_parser("ablate")
    subparsers.add_parser("profile")
    subparsers.add_parser("export")
    return parser


def main() -> None:
    args, remaining = build_parser().parse_known_args()
    sys.argv = [f"epnet-{args.command}", *remaining]
    if args.command == "train":
        train.main()
    elif args.command == "eval":
        evaluate.main()
    elif args.command == "infer":
        infer.main()
    elif args.command == "ablate":
        ablate.main()
    elif args.command == "profile":
        profile.main()
    elif args.command == "export":
        export.main()
    else:
        raise ValueError(f"Unsupported command: {args.command}")


__all__ = ["main"]
