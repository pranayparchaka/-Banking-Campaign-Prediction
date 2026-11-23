# Machine Learning and Application Workflow

This workflow documents the implemented path from `bank-full.csv` to the
dashboard and API results. It does not include deployment or public URL steps.

## Complete ML workflow

```mermaid
flowchart TD
    A[bank-full.csv] --> B[Data validation and EDA]
    B --> C[Separate target y]
    C --> D[Exclude duration from pre-call features]
    D --> E[Stratified development/test split]
    E --> F[Fit preprocessing on training partitions only]
    F --> G[Model comparison]

    G --> G1[Logistic Regression]
    G --> G2[Decision Tree]
    G --> G3[Random Forest]
    G --> G4[Gradient Boosting]

    G1 --> H[Evaluation]
    G2 --> H
    G3 --> H
    G4 --> H
    H --> I[Accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix]
    I --> J[Limited tuning using development data]
    J --> K[Tuned Gradient Boosting selected]
    K --> L[Fit selected model on development data]
    L --> M[5-fold development OOF raw probabilities]
    M --> N[Fit sigmoid and isotonic calibration mappings]
    N --> O[Sigmoid selected with ranking-preservation rule]
    O --> P[Calibrated model artifact]
    P --> Q[Final outer-test evaluation]
```

## Leakage controls

- `y` is kept strictly as the target.
- `duration` is excluded because it is observed after a campaign call.
- Preprocessing is fitted on training data, not on the full evaluation population.
- Hyperparameter tuning uses development data only.
- Calibration mappings use development-set out-of-fold raw probabilities.
- The outer test set is reserved for final evaluation.

## Preprocessing workflow

```mermaid
flowchart LR
    A[Feature frame] --> B[Identify numeric and categorical columns]
    B --> C[Numeric pipeline]
    B --> D[Categorical pipeline]
    C --> C1[Median imputation]
    C1 --> C2[StandardScaler]
    D --> D1[Most-frequent imputation]
    D1 --> D2[One-hot encoding]
    D2 --> D3[handle_unknown=ignore]
    C2 --> E[Combined transformed feature matrix]
    D3 --> E
```

## Probability calibration workflow

```mermaid
flowchart TD
    A[Development data] --> B[Stratified 5-fold split]
    B --> C[Clone base pipeline per fold]
    C --> D[Fit preprocessing and Gradient Boosting on fold-train rows]
    D --> E[Predict raw probabilities on fold-validation rows]
    E --> F[Complete OOF probability vector]
    F --> G[Fit sigmoid calibrator]
    F --> H[Fit isotonic calibrator]
    G --> I[Compare development OOF Brier and ranking metrics]
    H --> I
    I --> J[Sigmoid selected]
    J --> K[Evaluate once on untouched outer test set]
```

The final calibrated model uses the positive-class probability from the
sigmoid calibrator. The wrapper does not use the calibrator's hard class label
as a probability.

## Customer prioritization workflow

```mermaid
flowchart TD
    A[Future customer input] --> B[Pydantic schema validation]
    B --> C[Pre-call feature frame]
    C --> D[Underlying preprocessing pipeline]
    D --> E[Tuned Gradient Boosting]
    E --> F[Raw ranking score]
    F --> G[Sort descending]
    G --> H[Priority rank]
    H --> I[Very High / High / Medium / Low category]

    E --> F
    F --> J[Sigmoid calibration]
    J --> K[Calibrated response probability]
    K --> L[Response class and probability display]
```

The backend priority category is based on the calibrated probability using the
implemented thresholds:

- Very High: probability `>= 0.75`
- High: probability `>= 0.50`
- Medium: probability `>= 0.25`
- Low: probability `< 0.25`

The ordering itself uses the raw ranking score, not the category label.

## Capacity analysis workflow

```mermaid
flowchart TD
    A[Future customer list] --> B[Validate each customer]
    B --> C[Calculate raw ranking score]
    C --> D[Sort descending by raw score]
    D --> E[Calculate K from capacity percentage]
    E --> F[floor(N x capacity / 100)]
    F --> G[Select Top-K customers]
    G --> H[Read sigmoid-calibrated probabilities]
    H --> I[Average calibrated probability]
    H --> J[Sum calibrated probabilities]
    J --> K[Expected responders estimate]
```

At 100% capacity, all supplied customers are selected. For future customer
inputs, actual historical outcomes are unknown and are not returned by the API.

**Expected responders are estimates, not guaranteed subscriptions.**

