import os
import subprocess

def track_and_push(paths, description: str):
    """
    For each path in `paths`, do:
      1. dvc add --force <relpath> --file shared_volume/<relpath>.dvc
      2. git add shared_volume/<relpath>.dvc
    Then git commit & git push.
    """
    cwd = os.getcwd()
    # make sure shared_volume exists
    os.makedirs("shared_volume", exist_ok=True)

    dvc_files = []
    for p in paths:
        # e.g. p = "/app/models" or "/app/data/processed"
        rel = os.path.relpath(p, cwd)  # "models" or "data/processed"
        # target DVC-file under shared_volume
        dvc_file = os.path.join("shared_volume", f"{rel}.dvc")
        # 1) track with custom file location
        subprocess.run([
            "dvc", "add", "--force", rel,
            "--file", dvc_file
        ], check=True, text=True)
        # 2) stage that exact file
        subprocess.run(["git", "add", dvc_file], check=True, text=True)
        dvc_files.append(dvc_file)

    # commit & push once for all
    commit_msg = f"dvc: {description}"
    subprocess.run(["git", "commit", "-m", commit_msg], check=True, text=True)
    subprocess.run(["git", "push"], check=True, text=True)

    print(f"Tracked & pushed: {', '.join(dvc_files)}")
