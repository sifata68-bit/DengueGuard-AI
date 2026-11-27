# DengueGuard AI

**Team:** HackSync  
**Project:** EDU AI Engineering Hackathon 2026  
**Application:** https://dengueguard-ai.streamlit.app/  
**Repository:** https://github.com/sifata68-bit/DengueGuard-AI

DengueGuard AI is a ward-level dengue forecasting and operational decision-support application for Bangladesh. It forecasts dengue cases 2, 3, and 4 weeks ahead across 60 wards in five urban areas, then presents risk information and planning tools through a Streamlit dashboard.

The forecasting pipeline uses historical dengue surveillance, weather observations, rainfall lag features, and environmental indicators. The application also includes resource allocation and hospital surge planning tools, plus a vector-control simulation backend.

> **Important:** This is an educational hackathon decision-support project. Forecasts and scenario outputs are estimates, not official public-health alerts, medical advice, confirmed resource requirements, or instructions to transfer patients.

## Phase 2 integration status — 27 November 2026

The Phase 2 backend and frontend integration has been merged into the repository's `main` branch.

- Resource Allocation supports risk-prioritized distribution of saline bags and diagnostic kits from operator-provided inventory, demand assumptions, and city-level spraying-crew limits.
- Hospital Surge compares forecast-derived bed demand with operator-provided ward capacity and produces same-city overflow planning recommendations.
- The Vector Control simulation backend supports scenario-based intervention planning, constrained team assignment, estimated case reductions, and resource-savings calculations. The Streamlit page remains a preview and is not yet connected end-to-end to the simulation service.
- The Dispatch Center remains a preview; dispatch generation and external messaging are not operational.
- The full automated test suite passed **214 tests** at this integration checkpoint.
- The team confirmed that the local Streamlit application pages loaded during a manual browser smoke check. The deployed application was also reported working by the team.

These are checkpoint results, not a guarantee that future changes or every operational scenario will behave identically. Re-run tests and verify the deployment after subsequent changes.

## What the system does

### Dengue forecasting
- Generates ward-level forecasts for 2-, 3-, and 4-week horizons.
- Uses a separate locally trained model for each horizon.
- Combines historical dengue counts, weather, rainfall lags and rolling rainfall, environmental vulnerability, and calendar features.
- Classifies forecast risk as Low, Medium, High, or Critical.
- Displays nationwide, city-level, and ward-level information in the Streamlit application.

### Operational planning
- **Resource Allocation:** prioritizes wards and allocates available saline bags and diagnostic kits using supplied inventory and per-case demand assumptions. It also supports city-level spraying-crew limits.
- **Hospital Surge:** compares estimated bed demand with user-provided ward capacity, identifies overloaded wards, and creates same-city overflow routing recommendations.
- **Vector Control backend:** simulates a user-defined intervention scenario, allocates a limited number of teams by risk and population density, and estimates cases and resource use avoided. The current Streamlit page is still a preview and is not yet connected to this backend.
- **Dispatch Center:** currently a UI preview; dispatch generation is disabled.

The resource and hospital planning tools depend on user-supplied operational inputs. Scenario assumptions must not be presented as verified medical or public-health standards.

## Geographic and dataset coverage

The source dataset is `data/raw/hackathon_preli_dataset.xlsx`.

| Item | Coverage |
|---|---|
| Weekly records | 31,320 |
| Wards | 60 |
| Urban areas | 5 |
| Historical period | 2016–2025 |
| Forecast horizons | 2, 3, and 4 weeks |

The five urban areas are Chattogram, Dhaka North, Dhaka South, Gazipur, and Narayanganj.

The original data includes dengue case counts, temperature, rainfall, relative humidity, population density, waterlogging risk, and sanitation indicators. The cleaned dataset is stored under `data/processed/` and is generated locally from the raw source.

The latest observation in the supplied dataset is dated **29 December 2025**. The application does not automatically obtain live 2026 dengue surveillance or weather data.

## Technology

- Python 3.11
- Streamlit
- Pandas and NumPy
- scikit-learn
- Joblib
- Plotly
- pytest

Forecast inference uses locally stored Joblib models. No external weather, dengue-prediction, or LLM API is required for the local forecasting workflow.

## Repository layout

```text
DengueGuard-AI/
├── app.py
├── requirements.txt
├── README.md
├── ARCHITECTURE.md
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── production/
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── predict.py
│   ├── hospital_surge.py
│   ├── resource_allocation.py
│   ├── vector_control.py
│   └── vector_control_service.py
└── tests/
```

Some generated data and model artifacts are intentionally not tracked in Git. See `.gitignore` and the setup instructions below.

## Run locally (Windows PowerShell)

From the repository root:

1. Create a Python 3.11 virtual environment:

```powershell
py -3.11 -m venv venv
```

2. Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

4. Ensure the supplied raw dataset and production model artifacts are present in their expected locations. The processed dataset is generated by the preprocessing pipeline.

5. Start the application:

```powershell
python -m streamlit run app.py
```

Streamlit prints a local address, usually http://localhost:8501.

## Run tests

With the virtual environment activated:

```powershell
python -m pytest -q
```

At the 27 November 2026 Phase 2 integration checkpoint, the full suite completed with **214 passing tests**. Run the tests again after making changes; the count may change as the project evolves.

## Machine-learning workflow

The data is split chronologically to preserve time order:

| Split | Period | Purpose |
|---|---|---|
| Training | 2016–2021 | Fit candidate models |
| Validation | 2022–2023 | Model selection and tuning |
| Test | 2024–2025 | Held-out evaluation period |

The test period is kept separate from model selection. The production pipeline uses 25 engineered features, including current weather, rainfall lags and rolling values, static environmental features, calendar features, and historical dengue values. Historical dengue features use only earlier observations; current-week dengue cases are excluded from production features.

Separate HistGradientBoostingRegressor models are used for the 2-, 3-, and 4-week horizons.

### Validation results

| Horizon | MAE | RMSE |
|---|---:|---:|
| 2 weeks | 1.232 | 2.431 |
| 3 weeks | 1.361 | 2.602 |
| 4 weeks | 1.678 | 3.115 |

These are **validation regression metrics**, not classification accuracy and not test-set results. Do not convert them into an accuracy percentage.

## Risk categories

The four labels are Low, Medium, High, and Critical. Their thresholds are derived from training-period in-season nonzero case-count quantiles. They are project-specific model thresholds, **not official government or public-health alert thresholds**.

Predicted case counts refer to an individual future week, not a cumulative total.

## Operational assumptions and limitations

- Resource demand ratios must carry their source and provenance status. User-supplied or scenario-assumption values are not automatically verified standards.
- Hospital bed demand uses a configurable beds-per-case scenario ratio. The default is an assumption, not a clinically validated ratio.
- Hospital capacity must represent actual available beds supplied by the operator. Historical `hospital_beds_required` estimates are not hospital capacity.
- Overflow routing is a planning recommendation; it does not confirm that beds, transport, staff, or patient transfers are available.
- Vector-control efficacy is supplied as a scenario assumption. The simulator does not learn causal intervention effectiveness from intervention-outcome history.
- Risk thresholds and operational ratios require review by public-health and operational experts before real-world use.
- Forecast quality depends on historical data quality and coverage. There is no automatic live data feed in the current forecasting workflow.

## Team and development

DengueGuard AI is developed by **Team HackSync** for the EDU AI Engineering Hackathon 2026. Team contributions should be represented accurately by the repository's actual commit history; this README does not assign individual ownership of every file.

AI assistants were used as development aids for coding support, debugging, technical explanations, and documentation. Team members are responsible for understanding, reviewing, testing, and explaining the submitted implementation. AI use should be disclosed in accordance with the event rules.

## License and use

No license is specified here. Unless a license is added to the repository, do not assume the code is licensed for unrestricted reuse.
