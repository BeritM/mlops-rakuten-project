import os
import sys
import subprocess
from pathlib import Path

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parents[2] # parents[2] is the root of the project
plugins_path = project_root / "plugins"
docs_path = project_root / "docs"

env = os.environ.copy()
env["PYTHONPATH"] = str(plugins_path)

modules = [
    "cd4ml.data_processing.step01_combine_xy",
    "cd4ml.data_processing.step02_text_cleaning",
    "cd4ml.data_processing.step03_split_data",
    "cd4ml.data_processing.step04_tfidf_transform"
]

docs_path.mkdir(exist_ok=True)

for module in modules:
    print(f"Generating doc for: {module}")
    result = subprocess.run(
        [sys.executable, "-m", "pydoc", "-w", module],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"Failed to generate doc for {module}")
        print(result.stderr)
    else:
        print(result.stdout)

# Move HTML files
for html_file in project_root.glob("*.html"):
    if "data_processing" in html_file.name:
        target = docs_path / html_file.name
        if target.exists():
            print(f"Skipping move: {html_file.name} already exists in docs/")
        else:
            html_file.rename(target)
            print(f"Moved {html_file.name} /docs")