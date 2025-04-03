# Demonstrate how to use dvc within docker

1. Intsall uv with make uv
2. Add your dagshub credentials to the Makefile at the set-dvc key and run make set-dvc
3. Create a virtual environment with uv venv
4. Install all packages and dependencies with uv sync
5. Run the command below

'''
make all
''' 