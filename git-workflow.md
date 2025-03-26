# Git Branching Strategy for MLOps Rakuten Projects

This document outlines the branching workflow used in this MLOps project. It is inspired by the Azure MLOps Accelerator but adapted for general GitHub-based MLOps development.

Source:  
[Azure MLOps Accelerator – Branching Strategy](https://microsoft.github.io/azureml-ops-accelerator/4-Migrate/dstoolkit-mlops-base/docs/how-to/BranchingStrategy.html)

---

## Branch Overview

We use a simplified but robust branching strategy:

main ← production-ready (PROD) 
│ 
└── dev ← integration branch (DEV/TEST) 
├── dev-new-feature-<name> ← feature development 
└── dev-fix-bug-<name> ← bugfix development


---

## Branch Purpose

| Branch               | Purpose                                                                 |
|----------------------|-------------------------------------------------------------------------|
| `main`               | Stable, production-ready code. Only updated through Pull Requests.      |
| `dev`                | Integration branch for all tested features and bugfixes.                |
| `dev-new-feature-*`  | Feature development branches (created from `dev`).                      |
| `dev-fix-bug-*`      | Bugfix development branches (created from `dev`).                       |


---