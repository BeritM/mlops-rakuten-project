```mermaid
flowchart LR
  %% Environment setup
  subgraph "Environment setup"
    direction LR
    GH["Code Version Control<br>GitHub"]
    DS_env["Data Versioning & Storage<br>Dagshub + S3"]
    GA["CI/CD<br>GitHub Actions"]
    DOCK["Virtualization<br>docker"]
    AF["Automation<br>Airflow"]
    %% invisible connectors (still count as links 0–3)
    GH -.-> DS_env -.-> GA -.-> DOCK -.-> AF
  end
  %% hide those first four links
  linkStyle 0,1,2,3 opacity:0

  %% Data Management
  subgraph "Data Management"
    direction LR
    RAW[/Raw Product<br>Data/]
    FB[/Human<br>Feedback/]
    PRE[Preprocessing<br>Service]
    CONS[/Consolidated<br>Dataset/]
  end

  %% ML Pipeline Steps
  subgraph "Model Training & Validation"
    direction LR
    TRN[Model Training<br>Service]
    VAL[Model Validation<br>Service]
  end

  %% Model Tracking & Registry
  subgraph "Model Tracking & Registry"
    direction LR
    ML[Model Tracking<br>MLflow]
    REG[Model Registry<br>MLflow]
  end

  %% Deployment & Security
  subgraph "Deployment & Security"
    direction LR
    AUTH[Authentication Service<br>FastAPI]
    MODEL_ART[/Model Artifact/]
    API[Prediction Service<br>FastAPI]
  end

  %% Monitoring
  subgraph "Monitoring"
    direction LR
    EV[Model Monitoring<br>Evidently AI]
    PR[Performance Metrics<br>Prometheus]
    GF[Dashboards<br>Grafana]
  end

  %% Control & Data Flows
  RAW & FB --> PRE--> CONS --> TRN --> VAL

  TRN --> ML --> REG
  VAL --> ML

  REG --> MODEL_ART --> API
  API --- AUTH

  API --> FB
  API --> EV
  API --> PR --> GF
