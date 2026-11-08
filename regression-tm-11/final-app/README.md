# California Housing Price Prediction App (TM11 Regresi)

A clean, production-ready regression web application built with **Python**, **Streamlit**, and **XGBoost** to estimate California median house values based on census block characteristics.

---

## 1. Project Structure

```text
final-app/
├── app.py                 # Core Streamlit application
├── main.py                # Alternate entrypoint
├── utils.py               # Preprocessing, validation, and inference helpers
├── model/
│   ├── trained_model.pkl  # Trained XGBoost Regressor model
│   ├── scaler.pkl         # Fitted StandardScaler
│   └── ocean_encoder.pkl  # Fitted OrdinalEncoder
├── requirements.txt       # Project dependencies
└── README.md              # Documentation and execution guide
```

---

## 2. Requirements & Installation

Make sure Python 3.10+ is installed. Then install the necessary dependencies:

```bash
pip install -r requirements.txt
```

---

## 3. Running the Application

Execute the Streamlit application from this directory:

```bash
streamlit run app.py
```

Alternatively:

```bash
streamlit run main.py
```

The application will launch in your default web browser (typically at `http://localhost:8501`).

---

## 4. Application Architecture

1. **Overview Tab:**
   - Summarizes project scope, dataset origin, and regression objectives.
   - Outlines the selected model architecture (**XGBoost Regressor**) and hyperparameter tuning configuration.
   - Displays evaluation metrics (MAE, MSE, RMSE, MAPE, $R^2$) across training and testing splits.

2. **Price Prediction Tab:**
   - Multi-column form supporting all original census features (coordinates, housing age, room counts, population, income, coastal proximity).
   - Validates input values against domain-specific logical constraints before inference.
   - Automatically computes derived feature ratios (`rooms_per_household`, `bedrooms_per_room`, `population_per_household`) and scales inputs through the original pipeline.
   - Outputs the formatted prediction (`$XXX,XXX.XX`) alongside derived feature statistics.

3. **Model Information Tab:**
   - Outlines the XGBoost learning mechanism and loss regularization.
   - Details all 12 input features and transformations.
   - Documents the end-to-end preprocessing pipeline and known model limitations.

