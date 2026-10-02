"""Run the unmodified train.py on Apple MPS. Only the device whitelist in validate_args is relaxed."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import types  # noqa: E402
sys.modules.setdefault("wandb", types.ModuleType("wandb"))  # only used with --wandb
import train  # noqa: E402

_validate = train.validate_args


def validate_args(args):
    if args.device == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("MPS is unavailable")
        args.device = "cpu"            # pass the original checks...
        _validate(args)
        args.device = "mps"            # ...then train on MPS
    else:
        _validate(args)


train.validate_args = validate_args

if __name__ == "__main__":
    train.main(train.build_parser().parse_args())
