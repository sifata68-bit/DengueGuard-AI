DengueGuard AI

AI-powered urban dengue outbreak prediction and ward-level risk forecasting system for Bangladesh.

DengueGuard AI forecasts dengue cases 2, 3, and 4 weeks ahead across 60 wards in five urban areas. It combines historical dengue cases, weather patterns, rainfall lag features, and environmental vulnerability indicators to support ward-level risk monitoring.

FEATURES

- Ward-level dengue forecasts for 2, 3, and 4 weeks ahead.
- Multi-week rainfall lag and rolling rainfall features.
- Historical dengue case trend features using previous weeks only.
- Machine learning models trained separately for each forecast horizon.
- Four risk categories: Low, Medium, High, and Critical.
- Bangladesh-wide risk overview with city and horizon filters.
- Individual ward forecasts with weather and environmental indicators.
- Interactive Streamlit dashboard.
- Local model inference using saved Joblib models.

DATASET

The project uses the hackathon-provided historical dataset:

data/raw/hackathon_preli_dataset.xlsx

Dataset coverage:

- 31,320 weekly observations.
- 60 wards.
- Five city areas: Chattogram, Dhaka North, Dhaka South, Gazipur, and Narayanganj.
- Historical period: 2016–2025.

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
├── app.py
├── requirements.txt
├── README.md
├── ARCHITECTURE.md
├── CONTRACT.md
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── production/
├── src/
├── tests/
└── assets/
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
| Training | 2016–2021 | Model fitting |
| Validation | 2022–2023 | Model selection |
| Test | 2024–2025 | Reserved for final evaluation |

Separate HistGradientBoostingRegressor models are trained for each forecast horizon.

Production features include:

- Current-week weather conditions.
- Rainfall lags and rolling rainfall sums.
- Ward-level environmental vulnerability indicators.
- Calendar and seasonal features.
- Previous dengue case counts and rolling historical averages.

Current-week dengue cases are excluded from production features. Historical dengue features use only observations available before the forecast issue date.

Models are saved locally using Joblib for efficient inference.

See ARCHITECTURE.md for the system design, data flow, leakage prevention, and model selection details.

RISK CATEGORIES

| Category | Predicted Dengue Cases |
|---|---|
| Low | Below 3 |
| Medium | 3 to below 6 |
| High | 6 to below 10 |
| Critical | 10 or above |

These are provisional model-based thresholds derived from the training data. They are not official public-health alert standards.

TESTING

Run the project test suite:

```bash
python -m pytest -q
```

LIMITATIONS

- Forecasts depend on the quality and coverage of historical data.
- The latest dataset observation is dated 29 December 2025; forecasts are based on historical observations, not live 2026 weather or dengue reports.
- Risk thresholds require further validation with public-health experts.
- Model forecasts are estimates and should not replace official public-health guidance.

AI ASSISTANCE DISCLOSURE

AI assistance was used during project development for coding support, debugging, research, documentation, and technical guidance.

AI assistants used:

- ChatGPT by OpenAI
- Claude by Anthropic

The project developer reviewed, integrated, tested, and made the final decisions regarding the implementation.

DISCLAIMER

DengueGuard AI is an educational hackathon project. Its predictions are intended for research and demonstration purposes and should not be treated as official medical advice or public-health alerts.