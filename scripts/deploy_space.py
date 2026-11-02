"""Deploy the app to a Hugging Face Docker Space (free CPU tier).

Creates the Space if needed and uploads the code; Hugging Face then builds the Dockerfile.
The Space's README gets the YAML header Spaces requires, so the GitHub README stays clean.

Usage (from the project root, after `hf auth login` or with HF_TOKEN set):
    python -m scripts.deploy_space [--space neural-search]
"""

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

# Not imported from app.config so deploying only needs huggingface_hub, not torch.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

UPLOAD = ["app", "scripts", "results", "Dockerfile", ".dockerignore", "requirements.txt", "LICENSE"]

SPACE_HEADER = """---
title: Neural Search SciFact
emoji: 🔎
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Dense retrieval + cross-encoder reranking on BEIR SciFact
---

"""


def stage_files(target: Path, github_url: str | None) -> None:
    for name in UPLOAD:
        source = PROJECT_ROOT / name
        if source.is_dir():
            shutil.copytree(source, target / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(source, target / name)
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    source_line = f"Source code: {github_url}\n\n" if github_url else ""
    (target / "README.md").write_text(SPACE_HEADER + source_line + readme, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--space", default="neural-search", help="Space name (created under your account).")
    parser.add_argument("--github-url", default=os.getenv("GITHUB_REPO_URL"))
    args = parser.parse_args()

    api = HfApi(token=os.getenv("HF_TOKEN"))
    user = api.whoami()["name"]
    repo_id = f"{user}/{args.space}"

    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        stage_files(Path(tmp), args.github_url)
        api.upload_folder(
            repo_id=repo_id,
            repo_type="space",
            folder_path=tmp,
            commit_message="Deploy from GitHub",
            delete_patterns=["*"],  # mirror exactly: remove files deleted locally
        )

    print(f"Uploaded to https://huggingface.co/spaces/{repo_id}")
    print(f"App URL (live once the build finishes, ~5-10 min): https://{user.lower()}-{args.space}.hf.space")


if __name__ == "__main__":
    main()
