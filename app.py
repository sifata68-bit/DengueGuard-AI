
import streamlit as st
import pandas as pd
import plotly.express as px

from src.preprocessing import get_clean_data
from src.predict import predict_ward, predict_all_wards
from src.config import WARD_COL, CITY_COL, DATE_COL

st.set_page_config(
    page_title="DengueGuard AI",
    page_icon="🦟",
    layout="wide",
)


@st.cache_data
def load_data():
    return get_clean_data()


@st.cache_data
def load_all_ward_forecasts(clean_data):
    return predict_all_wards(clean_data)


st.title("🦟 DengueGuard AI")
st.subheader("AI-Powered Dengue Outbreak Prediction System")

st.markdown(
    """
    Monitor ward-level dengue risk using historical dengue cases,
    weather conditions, and environmental risk factors.
    """
)

try:
    with st.spinner("Loading cleaned dengue data..."):
        clean_data = load_data()

except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()


col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Dataset Records", f"{len(clean_data):,}")

with col2:
    st.metric(
        "Geographic Coverage",
        f"{clean_data[WARD_COL].nunique()} Wards",
    )

with col3:
    latest_date = pd.to_datetime(clean_data[DATE_COL]).max()
    st.metric("Latest Data", latest_date.strftime("%d %b %Y"))


st.divider()

st.header("Bangladesh-Wide Dengue Risk Overview")

st.caption(
    "Explore predicted dengue risk across all 60 wards. "
    "Filter predictions by city and forecast horizon."
)

try:
    with st.spinner("Generating all-ward forecasts..."):
        all_forecasts = load_all_ward_forecasts(clean_data)

    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:
        cities = ["All Bangladesh"] + sorted(
            all_forecasts[CITY_COL].dropna().unique().tolist()
        )

        selected_city = st.selectbox(
            "Filter by city",
            cities,
            key="overview_city",
        )

    with filter_col2:
        selected_horizon = st.selectbox(
            "Forecast horizon",
            [2, 3, 4],
            format_func=lambda x: f"{x} Weeks Ahead",
            key="overview_horizon",
        )

    overview = all_forecasts[
        all_forecasts["horizon_weeks"] == selected_horizon
    ].copy()

    if selected_city != "All Bangladesh":
        overview = overview[
            overview[CITY_COL] == selected_city
        ]

    risk_order = ["Low", "Medium", "High", "Critical"]

    risk_counts = (
        overview["risk_category"]
        .value_counts()
        .reindex(risk_order, fill_value=0)
    )

    metric_cols = st.columns(4)

    for col, risk in zip(metric_cols, risk_order):
        with col:
            st.metric(
                f"{risk} Risk Wards",
                int(risk_counts[risk]),
            )

    st.subheader("Ward Risk Distribution")

    risk_chart_data = (
        risk_counts
        .rename_axis("Risk Category")
        .reset_index(name="Number of Wards")
    )

    fig = px.bar(
        risk_chart_data,
        x="Risk Category",
        y="Number of Wards",
        color="Risk Category",
        category_orders={"Risk Category": risk_order},
        title=f"Dengue Risk Distribution — {selected_city}",
        text="Number of Wards",
    )

    fig.update_layout(
        showlegend=False,
        yaxis_title="Number of Wards",
        xaxis_title="Risk Category",
        yaxis_rangemode="tozero",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Ward-Level Risk Details")

    overview_table = overview[
        [
            WARD_COL,
            CITY_COL,
            "predicted_cases",
            "risk_category",
        ]
    ].copy()

    overview_table.columns = [
        "Ward",
        "City",
        "Predicted Cases",
        "Risk Category",
    ]

    overview_table = overview_table.sort_values(
        "Ward",
        ascending=True,
    ).reset_index(drop=True)

    st.caption(
        f"Showing {len(overview_table)} of "
        f"{all_forecasts['ward_id'].nunique()} wards"
        if selected_city == "All Bangladesh"
        else f"Showing {len(overview_table)} wards in {selected_city}"
    )

    st.dataframe(
        overview_table,
        use_container_width=True,
        height=600,
        hide_index=True,
    )

except Exception as exc:
    st.error(f"Could not generate the all-ward overview: {exc}")


st.divider()

st.header("Ward-Level Dengue Forecast")

wards = sorted(clean_data[WARD_COL].dropna().unique())

selected_ward = st.selectbox(
    "Select a ward",
    wards,
)

if st.button("Generate Forecast", type="primary"):

    try:
        with st.spinner("Generating forecasts..."):
            forecasts = predict_ward(
                clean_data=clean_data,
                ward_id=selected_ward,
            )

        st.success("Forecast generated successfully!")

        st.subheader("Dengue Forecast Results")

        city_name = (
            forecasts[CITY_COL].iloc[0]
            if CITY_COL in forecasts.columns
            else "Unknown"
        )

        forecast_date = pd.to_datetime(
            forecasts[DATE_COL].iloc[0]
        ).strftime("%d %b %Y")

        st.caption(
            f"Ward: {selected_ward} | "
            f"City: {city_name} | "
            f"Forecast issued using data up to: {forecast_date}"
        )

        st.subheader("Weather & Environmental Risk Drivers")

        selected_date = pd.to_datetime(
            forecasts[DATE_COL].iloc[0]
        )

        driver_row = clean_data[
            (clean_data[WARD_COL] == selected_ward)
            & (
                pd.to_datetime(clean_data[DATE_COL])
                == selected_date
            )
        ]

        if not driver_row.empty:
            drivers = driver_row.iloc[0]

            driver_col1, driver_col2, driver_col3 = st.columns(3)

            with driver_col1:
                st.metric(
                    "Average Temperature",
                    f"{drivers['avg_temp_c']:.1f} °C",
                )
                st.metric(
                    "Rainfall",
                    f"{drivers['rainfall_mm']:.1f} mm",
                )

            with driver_col2:
                st.metric(
                    "Relative Humidity",
                    f"{drivers['relative_humidity_pct']:.1f}%",
                )
                st.metric(
                    "Waterlogging Risk Index",
                    f"{drivers['water_logging_risk_index']:.2f}",
                )

            with driver_col3:
                st.metric(
                    "Population Density",
                    f"{drivers['population_density_sqkm']:,.1f} / km²",
                )
                st.metric(
                    "Sanitation Score",
                    f"{drivers['sanitation_score']:.2f}",
                )

        else:
            st.warning("Weather and environmental data unavailable.")

        cols = st.columns(3)

        for col, (_, row) in zip(cols, forecasts.iterrows()):
            with col:
                st.metric(
                    f"{row['horizon_weeks']}-Week Forecast",
                    f"{row['predicted_cases']:.2f} cases",
                )

                risk = row["risk_category"]

                if risk == "Low":
                    st.success(f"Risk: {risk}")
                elif risk == "Medium":
                    st.warning(f"Risk: {risk}")
                else:
                    st.error(f"Risk: {risk}")

        st.subheader("Forecast Details")

        display_df = forecasts[
            [
                "horizon_weeks",
                "predicted_cases",
                "risk_category",
            ]
        ].copy()

        display_df.columns = [
            "Forecast Horizon (Weeks)",
            "Predicted Cases",
            "Risk Category",
        ]

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Forecast Trend")

        chart_data = forecasts.copy()

        chart_data["Forecast Horizon"] = (
            chart_data["horizon_weeks"].astype(str) + " Weeks"
        )

        fig = px.bar(
            chart_data,
            x="Forecast Horizon",
            y="predicted_cases",
            text="predicted_cases",
            title="Predicted Dengue Cases by Forecast Horizon",
            labels={
                "predicted_cases": "Predicted Cases",
                "Forecast Horizon": "Forecast Period",
            },
        )

        fig.update_traces(
            texttemplate="%{text:.2f}",
            textposition="outside",
        )

        fig.update_layout(
            showlegend=False,
            yaxis_title="Predicted Dengue Cases",
            xaxis_title="Forecast Horizon",
            yaxis_rangemode="tozero",
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as exc:
        st.error(f"Forecast generation failed: {exc}")


st.divider()

st.caption(
    "DengueGuard AI | EDU AI Engineering Hackathon 2026"
)

st.caption(
    "Forecasts are model estimates, not official public-health alerts."
)
