# Using DVC Within Docker: A Step-by-Step Guide

This guide demonstrates how to set up and run DVC within Docker containers using a Makefile to automate common tasks. Follow the steps below to get started.

## Overview of Steps

1. **Install uv**  
   Use the `make uv` command to install the uv tool (from Astral).

2. **Configure Dagshub Credentials for DVC**  
   Add your Dagshub credentials to the Makefile under the `set-dvc` target and run:
   ```bash
   make set-dvc
