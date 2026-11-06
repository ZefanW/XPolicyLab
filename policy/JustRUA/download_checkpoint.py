"""Fetch and verify the Depth Pro weights used by JustRUA."""
import argparse
from pathlib import Path

from XPolicyLab.policy.JustRUA.observations.depth import CHECKPOINT_FILES, MODEL, REVISION, verify_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="local Depth Pro checkpoint directory")
    args = parser.parse_args()
    from huggingface_hub import snapshot_download

    directory = args.directory.expanduser().resolve()
    snapshot_download(MODEL, revision=REVISION, allow_patterns=list(CHECKPOINT_FILES), local_dir=directory)
    verify_checkpoint(directory)
    print(f"Verified {MODEL} at {directory}")


if __name__ == "__main__":
    main()
