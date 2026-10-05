"""Publish the static web client to Hugging Face Spaces."""

import os
import sys
import argparse
from pathlib import Path
from huggingface_hub import HfApi

SPACE_DIR = Path(__file__).resolve().parent.parent / "space"
DEFAULT_SPACE_ID = "ghananlpcommunity/ghana-chat"


def main():
    parser = argparse.ArgumentParser(description="Publish Ghana Chat to Hugging Face Spaces")
    parser.add_argument("--repo-id", default=DEFAULT_SPACE_ID, help="HF Space repo id (e.g. ghananlpcommunity/ghana-chat)")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="Hugging Face API token")
    args = parser.parse_args()

    api = HfApi(token=args.token)
    print(f"Creating/verifying HF Space: {args.repo_id}...")
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="space",
        space_sdk="static",
        exist_ok=True,
        private=False
    )

    print(f"Uploading files from {SPACE_DIR} to Space {args.repo_id}...")
    api.upload_folder(
        folder_path=str(SPACE_DIR),
        repo_id=args.repo_id,
        repo_type="space"
    )

    print(f"\nSuccessfully published! View live space at:")
    print(f"https://huggingface.co/spaces/{args.repo_id}")


if __name__ == "__main__":
    main()
