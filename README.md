# Nomophobia Score Predictor

Predict smartphone addiction severity using machine learning. Enter your age, usage time, symptoms, and behaviors to get an instant assessment.

## Quick Start

### Install
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run Web App
```bash
# Activate virtual environment (if not already active)
source .venv/bin/activate

# Start the app
streamlit run app.py
```
Open **http://localhost:8501** and fill out the form to get your nomophobia score.

## Project Files

- **app.py** — Interactive web application
- **Nomophobia_Score_Predictor.ipynb** — Data analysis & model training
- **Nomophobia.csv** — Dataset (605 MJPRU student responses)
- **requirements.txt** — Dependencies

## Tech Stack

- Python 3.9+
- Streamlit 1.27.0 (web app)
- scikit-learn 1.3.0 (ML models)
- pandas, numpy, matplotlib

## Score Interpretation

| Range | Risk Level |
|-------|-----------|
| 0-20 | Low |
| 20-30 | Moderate |
| 30-40 | High |
| 40+ | Very High |

## Data

Survey data from MJPRU students analyzing smartphone usage patterns and behavioral indicators. 605 responses total.
