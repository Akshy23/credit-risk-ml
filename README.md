# Credit Default Risk Prediction

An educational tabular machine-learning project that estimates whether a loan record is associated with default (`loan_status = 1`) or no default (`loan_status = 0`). It covers exploratory data analysis, preprocessing, model comparison, evaluation, model saving, and a small FastAPI prediction endpoint.

This is a learning project, not a lending or credit decision system.

## Dataset

The project uses the [Credit Risk Dataset on Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset). The dataset page lists the data as **CC0: Public Domain**. The CSV is not included in this repository. Download `credit_risk_dataset.csv` and place it here:

```text
data/credit_risk_dataset.csv
```

The notebook expects the target column `loan_status`, where `1` means default and `0` means no default.

## Project contents

```text
credit-risk-ml/
├── data/                         # Local dataset; excluded from Git
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── app.py                    # FastAPI app
│   └── credit_default_random_forest.joblib  # Generated locally; excluded from Git
├── requirements.txt
├── .gitignore
└── README.md
```

## Approach

- Used an 80/20 stratified train/test split.
- Built preprocessing into scikit-learn pipelines: median imputation and scaling for numeric fields, and most-frequent imputation plus one-hot encoding for categorical fields.
- Compared Logistic Regression, a Decision Tree, and a Random Forest on the same validation split.
- Excluded `loan_grade` and `loan_int_rate` from the first model because this project treats prediction as happening before final loan pricing.
- Selected a Random Forest and used a 0.30 probability threshold for the project demonstration.
- Saved the fitted preprocessing/model pipeline, feature list, and threshold together in a local Joblib artifact.

## Final test-set results

Random Forest at threshold 0.30:

| Metric | Result |
|---|---:|
| Accuracy | 87.5% |
| Default precision | 74.7% |
| Default recall | 64.6% |
| ROC-AUC | 0.872 |

Confusion matrix (actual rows, predicted columns; class order `0`, `1`):

```text
[[4783, 312],
 [ 503, 919]]
```

The threshold is an educational choice made to demonstrate the precision/recall tradeoff. It is not a recommended real-world credit policy.

## Run locally

Use Python 3.10 or newer. From the repository root, create and activate a virtual environment, then install the packages:

```bash
python -m venv .venv
source .venv/Scripts/activate  # Git Bash on Windows
python -m pip install -r requirements.txt
```

In VS Code, open `notebooks/01_data_understanding.ipynb`, select the `.venv` Python environment, and run the notebook cells from top to bottom. Keep the notebook's working directory set to `notebooks/`; the final cells train the Random Forest and create `credit_default_random_forest.joblib` beside `app.py`.

Start the API from the `notebooks/` directory:

```bash
python -m uvicorn app:app --reload
```

Open `http://127.0.0.1:8000/docs`, expand **POST `/predict`**, and submit JSON such as:

```json
{
  "person_age": 30,
  "person_income": 60000,
  "person_home_ownership": "RENT",
  "person_emp_length": 5,
  "loan_intent": "EDUCATION",
  "loan_amnt": 10000,
  "loan_percent_income": 0.17,
  "cb_person_default_on_file": "N",
  "cb_person_cred_hist_length": 5
}
```

The API returns a default probability, the label produced by the 0.30 threshold, and the threshold used.

## Limitations

- This dataset and model are for learning; they have not been validated for real credit decisions.
- No fairness, calibration, regulatory, or real-world deployment review has been performed.
- The 0.30 threshold reflects a classroom tradeoff, not a business or policy requirement.
- The locally generated `.joblib` artifact is intentionally excluded. Recreate it by running the notebook, and only load model artifacts you trust.

## License

The Kaggle dataset page lists the dataset as CC0. No license has been selected for this repository's code yet; add a `LICENSE` file before inviting others to reuse the code.
