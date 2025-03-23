import os
import subprocess
from pathlib import Path

# Locate directories
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parents[2]
output_dir = current_dir / "docs"

# Base module path for pydoc
base_module_path = "plugins.cd4ml.data_processing"

# Make sure output folder exists
output_dir.mkdir(exist_ok=True)

# Generate docs
for file in current_dir.glob("step*.py"):
    module_name = file.stem
    full_module = f"{base_module_path}.{module_name}"
    print(f"Generating documentation for: {full_module}")
    subprocess.run(
        ["python", "-m", "pydoc", "-w", full_module],
        cwd=project_root,  # this makes sure pydoc sees your modules
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

# Move generated .html files from root to docs/
for html_file in project_root.glob("*.html"):
    if "data_processing" in html_file.name:
        target = output_dir / html_file.name
        html_file.rename(target)

print("Documentation files saved in the 'docs/' folder.")
