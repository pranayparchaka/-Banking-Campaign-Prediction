# Presentation Diagram Set

This page collects the concise Mermaid diagrams suitable for a college project
report or presentation. They describe implemented functionality only.

## 1. End-to-end system view

```mermaid
flowchart TD
    User --> Dashboard[HTML/CSS/JavaScript Dashboard]
    Dashboard --> Client[Fetch API Client]
    Client --> FastAPI[FastAPI REST API]
    FastAPI --> Routes[Routes and Pydantic Schemas]
    Routes --> Service[ML Service]
    Routes --> Data[Data Service]
    Service --> Model[Calibrated Model Artifact]
    Model --> Base[Preprocessing + Tuned Gradient Boosting]
    Model --> Cal[Sigmoid Calibration]
    Base --> Raw[Raw ranking score]
    Cal --> Probability[Calibrated response probability]
    Raw --> Rank[Priority ranking]
    Probability --> Estimate[Expected responder estimate]
    Rank --> Results[API and dashboard results]
    Estimate --> Results
```

## 2. Model design

```mermaid
flowchart LR
    Dataset[bank-full.csv] --> Contract[Pre-call feature contract]
    Contract --> Numeric[Numeric imputation + scaling]
    Contract --> Categorical[Categorical imputation + one-hot encoding]
    Numeric --> Pipeline[Combined preprocessing pipeline]
    Categorical --> Pipeline
    Pipeline --> GB[Tuned Gradient Boosting]
    GB --> Raw[Raw ranking score]
    Raw --> Sigmoid[LogisticRegression sigmoid calibration]
    Sigmoid --> Calibrated[Calibrated response probability]
```

## 3. Prioritization and calibration distinction

```mermaid
flowchart TD
    Input[Customer input] --> Base[Base model]
    Base --> Raw[Raw ranking score]
    Raw --> Sort[Sort descending]
    Sort --> Rank[Priority rank]
    Raw --> Calibrate[Sigmoid calibrator]
    Calibrate --> Probability[Calibrated probability]
    Probability --> Category[Priority category]
    Probability --> Expected[Expected responders]
```

The two paths are intentionally separate:

| Value | Implemented use |
|---|---|
| Raw ranking score | Customer ordering and Top-K selection |
| Calibrated probability | Response probability, priority category, and expected responders |

## 4. Capacity decision flow

```mermaid
flowchart LR
    Customers[Customer list] --> Rank[Raw-score ranking]
    Rank --> K[Top-K from capacity percentage]
    K --> Cal[Calibrated probabilities]
    Cal --> Sum[Sum probabilities]
    Sum --> Estimate[Expected responders estimate]
```

The implemented Top-K rule is:

```text
K = floor(N × capacity_percentage / 100)
```

with 100% selecting all available customers. Small non-zero capacities select
at least one customer in the backend implementation.

## 5. Future-customer boundary

```mermaid
flowchart TD
    Future[Future customer] --> Features[Pre-call attributes only]
    Features --> Prediction[Prediction and prioritization]
    Prediction --> Outputs[Raw score + calibrated probability]
    Historical[Historical actual outcome y] -. retrospective training/evaluation only .-> Evaluation[Offline evaluation]
    Future -. no actual outcome available .-> Outputs
```

Actual outcomes are not available for future API requests. Expected responders
are model estimates, not guaranteed subscriptions.

