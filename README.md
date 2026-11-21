DengueGuard AI

AI-powered urban dengue outbreak prediction and ward-level risk forecasting system for Bangladesh.

DengueGuard AI forecasts dengue cases 2, 3, and 4 weeks ahead across 60 wards in five urban areas. It combines historical dengue cases, weather patterns, rainfall lag features, and environmental vulnerability indicators to support ward-level risk monitoring.

FEATURES

- Ward-level dengue forecasts for 2, 3, and 4 weeks ahead.
- Multi-week rainfall lag and rolling rainfall features.
- Historical dengue case trend features using previous weeks only.
- Separate machine learning models for each forecast horizon.
- Four risk categories: Low, Medium, High, and Critical.
- Bangladesh-wide risk overview with city and forecast horizon filters.
- Individual ward forecasts with weather and environmental indicators.
- Interactive Streamlit dashboard with dark and light themes.
- English and Bangla interface support.
- Local model inference using saved Joblib models.

DATASET

The project uses the hackathon-provided historical dataset:

`data/raw/hackathon_preli_dataset.xlsx`

Dataset coverage:

- 31,320 weekly observations.
- 60 wards.
- Five city areas: Chattogram, Dhaka North, Dhaka South, Gazipur, and Narayanganj.
- Historical period: 2016-2025.

The dataset includes dengue cases, temperature, rainfall, relative humidity, population density, waterlogging risk, and sanitation indicators.

TECHNOLOGY STACK

- Python 3.11
- Pandas and NumPy
- Scikit-learn
- Joblib
- Streamlit
- Plotly
- Pytest

PROJECT STRUCTURE

```text
DengueGuard-AI/
|-- app.py
|-- requirements.txt
|-- README.md
|-- ARCHITECTURE.md
|-- CONTRACT.md
|-- data/
|   |-- raw/
|   |-- processed/
|-- models/
|   |-- production/
|-- src/
|-- tests/
|-- assets/
```

SETUP AND INSTALLATION

1. Clone the repository

```bash
git clone https://github.com/sifata68-bit/DengueGuard-AI.git
cd DengueGuard-AI
```

2. Create a virtual environment

```bash
py -3.11 -m venv venv
```

3. Activate the environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

RUN THE APPLICATION

From the project root, with the virtual environment activated:

```bash
streamlit run app.py
```

Streamlit will provide a local URL to open the dashboard in your browser.

The application uses the locally stored dataset and trained models. No external API key is required for local prediction.

MODEL DEVELOPMENT

The dataset is split chronologically:

| Split | Period | Purpose |
|---|---|---|
| Training | 2016-2021 | Model fitting |
| Validation | 2022-2023 | Model selection |
| Test | 2024-2025 | Reserved for final evaluation |

Separate HistGradientBoostingRegressor models are trained for each forecast horizon.

Production features include:

- Current-week weather conditions.
- Rainfall lags and rolling rainfall sums.
- Ward-level environmental vulnerability indicators.
- Calendar and seasonal features.
- Previous dengue case counts and rolling historical averages.

Current-week dengue cases are excluded from production features. Historical dengue features use only observations available before the forecast issue date.

Models are saved locally using Joblib for efficient inference.

See `ARCHITECTURE.md` for the system design, data flow, chronological splitting, leakage prevention, and model selection details.

RISK CATEGORIES

| Category | Predicted Dengue Cases |
|---|---|
| Low | Below 3 |
| Medium | 3 to below 6 |
| High | 6 to below 10 |
| Critical | 10 or above |

These provisional model-based thresholds were derived from non-zero dengue case counts during the June-October dengue season in the training period (2016-2021). They are not official government or public-health alert thresholds.

Predicted cases represent an individual future week, not cumulative cases.

TESTING

Run the project test suite:

```bash
python -m pytest -q
```

The project includes tests for preprocessing, feature engineering, rainfall lag calculations, leakage prevention, model predictions, and other pipeline components.

GIT HISTORY & DEVELOPMENT PROCESS

DengueGuard AI was developed as a solo project during the EDU AI Engineering Hackathon 2026.

The Git repository maintains a chronological development history covering the major stages of implementation, including:

- Initial project setup and repository integration.
- Data analysis, forecasting targets, and baseline evaluation.
- Data cleaning, validation, and preprocessing.
- Multi-week rainfall lag and rolling feature engineering.
- Leakage checks and diagnostic model comparisons.
- Production model training and prediction pipeline.
- Interactive dashboard implementation.
- System architecture and project documentation.

Development work was organized through feature branches and merged pull requests, including the data-cleaning, ML-pipeline, and rainfall-lag work.

The repository's commit history is available at:

https://github.com/sifata68-bit/DengueGuard-AI/commits/main

No additional team members or distributed contributors are claimed; the project was developed by one developer.

AI ASSISTANTS & EXTERNAL API DISCLOSURE

AI assistants used during development:

- ChatGPT - OpenAI: Coding assistance, debugging, technical explanations, documentation, and development guidance.
- Claude - Anthropic: Additional AI-assisted development and technical support.

The developer reviewed, integrated, tested, and made the final implementation decisions.

EXTERNAL API USAGE

No external APIs are integrated into the application's forecasting workflow.

| Service | Usage |
|---|---|
| External weather API | Not used |
| External dengue prediction API | Not used |
| External LLM API | Not used |
| Third-party inference API | Not used |

The project uses the provided historical dataset, locally trained machine learning models, and local Joblib-based inference.

This design avoids unnecessary API requests, external inference costs, and network dependencies during prediction.

LIMITATIONS

- Forecasts depend on the quality and coverage of historical data.
- The latest dataset observation is dated 29 December 2025; forecasts are based on historical observations, not live 2026 weather or dengue reports.
- Risk thresholds require further validation with public-health experts.
- Model forecasts are estimates and should not replace official public-health guidance.

DISCLAIMER

DengueGuard AI is an educational hackathon project. Its predictions are intended for research and demonstration purposes and should not be treated as official medical advice or public-health alerts.