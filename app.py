import streamlit as st
import pandas as pd
import plotly.express as px

from src.preprocessing import get_clean_data
from src.predict import predict_ward, predict_all_wards
from src.config import WARD_COL, CITY_COL, DATE_COL
from src.resource_allocation import allocate_resources
from src.hospital_surge import (
    DEFAULT_BEDS_PER_CASE,
    assess_hospital_surge,
    generate_overflow_plan,
)

st.set_page_config(
    page_title="DengueGuard AI",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

if "language" not in st.session_state:
    st.session_state.language = "English"

theme = st.session_state.theme
language = st.session_state.language
dark = theme == "Dark"

T = {
    "English": {
        "dashboard": "Dashboard",
        "ward_forecast": "Ward Forecast",
        "settings": "Preferences",
        "theme": "Appearance",
        "language": "Language",
        "dark": "Dark",
        "light": "Light",
        "overview": "Bangladesh-Wide Dengue Risk Overview",
        "overview_desc": "Monitor predicted dengue risk across the country.",
        "records": "Dataset Records",
        "wards": "Geographic Coverage",
        "latest": "Latest Data",
        "date": "Forecast Reference Date",
        "day": "Day",
        "month": "Month",
        "year": "Year",
        "selected_date": "Selected date",
        "available_date": "Forecast data available up to",
        "city": "Filter by city",
        "all": "All Bangladesh",
        "horizon": "Forecast Horizon",
        "low": "Low Risk",
        "medium": "Medium Risk",
        "high": "High Risk",
        "critical": "Critical Risk",
        "distribution": "Ward Risk Distribution",
        "details": "Ward-Level Risk Details",
        "ward": "Ward",
        "predicted": "Predicted Cases",
        "risk": "Risk Category",
        "select_ward": "Select a ward",
        "generate": "Generate Forecast",
        "drivers": "Weather & Environmental Drivers",
        "temperature": "Average Temperature",
        "rainfall": "Rainfall",
        "humidity": "Relative Humidity",
        "waterlogging": "Waterlogging Risk Index",
        "density": "Population Density",
        "sanitation": "Sanitation Score",
        "results": "Forecast Results",
        "trend": "Forecast Trend",
        "issued": "Forecast issued using data up to",
        "forecast_details": "Forecast Details",
        "horizon_weeks": "Forecast Horizon (Weeks)",
        "cases": "Predicted Dengue Cases",
        "period": "Forecast Period",
        "loading": "Loading dengue data...",
        "generating": "Generating forecasts...",
        "success": "Forecast generated successfully!",
        "disclaimer": "Forecasts are model estimates, not official public-health alerts.",
        "intro": "An intelligent early-warning dashboard for ward-level dengue risk monitoring across Bangladesh.",
        "invalid_date": "This date does not exist. Please select a valid date.",
        "no_date": "No dataset date is available on or before your selection.",
        "unavailable": "Weather and environmental data unavailable.",
        "search": "Explore historical risk patterns and future dengue forecasts.",
        "navigation": "NAVIGATION",
        "data_period": "DATA PERIOD",
        "system": "SYSTEM",
        "showing": "Showing",
        "of": "of",
        "city_name": "City",
        "forecast_page_desc": "Generate multi-week dengue forecasts for an individual ward.",
        "risk_note": "Risk thresholds are model-derived and should not be interpreted as official health authority classifications.",
        "ward_detail_desc": "Ward-by-ward predicted cases and risk classification",
        "forecast_detail_desc": "A clear breakdown of each forecast horizon and its estimated risk",
        "risk_guide_title": "How Risk Levels Are Calculated",
        "risk_guide_desc": "Each forecast is classified using its predicted dengue cases for one future week.",
        "risk_guide_low": "Fewer than 3 predicted cases",
        "risk_guide_medium": "3 to fewer than 6 predicted cases",
        "risk_guide_high": "6 to fewer than 10 predicted cases",
        "risk_guide_critical": "10 or more predicted cases",
        "risk_guide_method": "These project-specific thresholds were derived from non-zero dengue case counts during the June–October dengue season in the training period (2016–2021). They are not official government or public-health alert thresholds.",
        "risk_guide_unit": "Predicted cases in a single future week · not cumulative",
        "resource_allocation": "Resource Allocation",
        "hospital_surge": "Hospital Surge",
        "vector_simulator": "Vector Control Simulator",
        "dispatch_center": "Dispatch Center",
        "resource_desc": "Review regional supply settings and prepare ward-level resource allocations.",
        "hospital_desc": "Review hospital bed demand, capacity, utilization and same-city overflow routing.",
        "vector_desc": "Explore planned vector-control interventions and their projected downstream effects.",
        "dispatch_desc": "Prepare ward-specific operational notices for field teams and local officers.",
        "prototype_notice": "Frontend preview only. Backend calculations are not connected yet; no operational results are being generated.",
        "regional_stock": "Available Regional Stock",
        "spraying_crews": "Total Available Spraying Crews",
        "stock_preview": "Stock availability preview",
        "crew_preview": "Crew capacity preview",
        "saline": "Saline Bags",
        "diagnostic_kits": "Diagnostic Kits",
        "beds_required": "Hospital Beds Required",
        "spraying_assignments": "Spraying Crew Assignments",
        "awaiting_backend": "Awaiting backend integration",
        "hospital_capacity": "Hospital Capacity Overview",
        "bed_demand": "Bed Demand",
        "available_beds": "Available Beds",
        "city_utilization": "City-wide Utilization",
        "ward_deficits": "Ward-level Bed Deficits",
        "overflow_routes": "Overflow Transfer Routes",
        "select_wards": "Select wards for intervention",
        "intervention_teams": "Available Vector-Control Teams",
        "simulation_horizon": "Simulation Horizon",
        "baseline_cases": "Baseline Cases",
        "simulated_cases": "Simulated Cases",
        "cases_prevented": "Estimated Cases Prevented",
        "dispatch_ward": "Select a ward",
        "dispatch_format": "Dispatch Format",
        "dispatch_preview": "Dispatch Document Preview",
        "generate_dispatch": "Generate Dispatch Document",
        "dispatch_unavailable": "Dispatch generation will be enabled after the export module is integrated.",
        "no_backend_results": "No calculated results are available yet.",
        "ui_control_note": "This control is a frontend preview and does not change resource allocations yet.",
        "select_channel": "Choose an output format"
    },
    "বাংলা": {
        "dashboard": "ড্যাশবোর্ড",
        "ward_forecast": "ওয়ার্ড পূর্বাভাস",
        "settings": "পছন্দসমূহ",
        "theme": "থিম",
        "language": "ভাষা",
        "dark": "ডার্ক",
        "light": "লাইট",
        "overview": "বাংলাদেশব্যাপী ডেঙ্গু ঝুঁকি পর্যবেক্ষণ",
        "overview_desc": "সারা দেশের ওয়ার্ডভিত্তিক ডেঙ্গু ঝুঁকি পর্যবেক্ষণ করুন।",
        "records": "ডেটাসেট রেকর্ড",
        "wards": "ওয়ার্ড সংখ্যা",
        "latest": "সর্বশেষ ডেটা",
        "date": "পূর্বাভাসের রেফারেন্স তারিখ",
        "day": "দিন",
        "month": "মাস",
        "year": "বছর",
        "selected_date": "নির্বাচিত তারিখ",
        "available_date": "সর্বশেষ উপলব্ধ ডেটা",
        "city": "শহর নির্বাচন",
        "all": "সমগ্র বাংলাদেশ",
        "horizon": "পূর্বাভাসের সময়সীমা",
        "low": "কম ঝুঁকি",
        "medium": "মাঝারি ঝুঁকি",
        "high": "উচ্চ ঝুঁকি",
        "critical": "অতি উচ্চ ঝুঁকি",
        "distribution": "ওয়ার্ডভিত্তিক ঝুঁকি বণ্টন",
        "details": "ওয়ার্ডভিত্তিক ঝুঁকির বিবরণ",
        "ward": "ওয়ার্ড",
        "predicted": "পূর্বাভাসকৃত আক্রান্ত",
        "risk": "ঝুঁকির স্তর",
        "select_ward": "ওয়ার্ড নির্বাচন করুন",
        "generate": "পূর্বাভাস তৈরি করুন",
        "drivers": "আবহাওয়া ও পরিবেশগত উপাদান",
        "temperature": "গড় তাপমাত্রা",
        "rainfall": "বৃষ্টিপাত",
        "humidity": "আপেক্ষিক আর্দ্রতা",
        "waterlogging": "জলাবদ্ধতার ঝুঁকি",
        "density": "জনসংখ্যার ঘনত্ব",
        "sanitation": "স্যানিটেশন স্কোর",
        "results": "পূর্বাভাসের ফলাফল",
        "trend": "পূর্বাভাসের প্রবণতা",
        "issued": "যে তারিখের ডেটা ব্যবহার করা হয়েছে",
        "forecast_details": "পূর্বাভাসের বিবরণ",
        "horizon_weeks": "পূর্বাভাসের সময়সীমা (সপ্তাহ)",
        "cases": "পূর্বাভাসকৃত ডেঙ্গু আক্রান্ত",
        "period": "পূর্বাভাসের সময়",
        "loading": "ডেঙ্গু ডেটা লোড হচ্ছে...",
        "generating": "পূর্বাভাস তৈরি হচ্ছে...",
        "success": "পূর্বাভাস সফলভাবে তৈরি হয়েছে!",
        "disclaimer": "এই পূর্বাভাস মডেলের অনুমান, সরকারি স্বাস্থ্য সতর্কতা নয়।",
        "intro": "বাংলাদেশের ওয়ার্ডভিত্তিক ডেঙ্গু ঝুঁকি পর্যবেক্ষণের জন্য একটি বুদ্ধিমান আগাম সতর্কতা ড্যাশবোর্ড।",
        "invalid_date": "এই তারিখটি সঠিক নয়। অনুগ্রহ করে বৈধ তারিখ নির্বাচন করুন।",
        "no_date": "নির্বাচিত তারিখের আগে কোনো ডেটা পাওয়া যায়নি।",
        "unavailable": "আবহাওয়া ও পরিবেশগত ডেটা পাওয়া যায়নি।",
        "search": "অতীতের ঝুঁকি এবং ভবিষ্যৎ ডেঙ্গু পূর্বাভাস পর্যবেক্ষণ করুন।",
        "navigation": "নেভিগেশন",
        "data_period": "ডেটার সময়সীমা",
        "system": "সিস্টেম",
        "showing": "দেখানো হচ্ছে",
        "of": "মোট",
        "city_name": "শহর",
        "forecast_page_desc": "একটি নির্দিষ্ট ওয়ার্ডের জন্য একাধিক সপ্তাহের ডেঙ্গু পূর্বাভাস তৈরি করুন।",
        "risk_note": "ঝুঁকির সীমা মডেল থেকে নির্ধারিত; এগুলো সরকারি স্বাস্থ্য কর্তৃপক্ষের শ্রেণিবিভাগ নয়।",
        "ward_detail_desc": "প্রতিটি ওয়ার্ডের পূর্বাভাসকৃত আক্রান্ত ও ঝুঁকির শ্রেণিবিভাগ",
        "forecast_detail_desc": "প্রতিটি পূর্বাভাসের সময়সীমা ও সম্ভাব্য ঝুঁকির বিস্তারিত বিবরণ",
        "risk_guide_title": "ঝুঁকির স্তর কীভাবে নির্ধারণ করা হয়",
        "risk_guide_desc": "প্রতিটি পূর্বাভাসের ক্ষেত্রে পরবর্তী একটি সপ্তাহের সম্ভাব্য ডেঙ্গু আক্রান্তের সংখ্যা ব্যবহার করে ঝুঁকি নির্ধারণ করা হয়।",
        "risk_guide_low": "৩ জনের কম সম্ভাব্য আক্রান্ত",
        "risk_guide_medium": "৩ থেকে ৬ জনের কম সম্ভাব্য আক্রান্ত",
        "risk_guide_high": "৬ থেকে ১০ জনের কম সম্ভাব্য আক্রান্ত",
        "risk_guide_critical": "১০ বা তার বেশি সম্ভাব্য আক্রান্ত",
        "risk_guide_method": "এই প্রকল্পের ঝুঁকির সীমাগুলো ২০১৬–২০২১ সালের প্রশিক্ষণ ডেটার জুন–অক্টোবর ডেঙ্গু মৌসুমে শূন্যের বেশি আক্রান্তের সংখ্যা থেকে নির্ধারণ করা হয়েছে। এগুলো সরকারি বা জনস্বাস্থ্য কর্তৃপক্ষের আনুষ্ঠানিক সতর্কতার সীমা নয়।",
        "risk_guide_unit": "একটি ভবিষ্যৎ সপ্তাহের সম্ভাব্য আক্রান্ত · মোট বা ক্রমযোজিত সংখ্যা নয়",
        "resource_allocation": "সম্পদ বরাদ্দ",
        "hospital_surge": "হাসপাতালের চাপ",
        "vector_simulator": "ভেক্টর নিয়ন্ত্রণ সিমুলেটর",
        "dispatch_center": "ডিসপ্যাচ সেন্টার",
        "resource_desc": "আঞ্চলিক সরবরাহের সেটিংস এবং ওয়ার্ডভিত্তিক সম্পদ বরাদ্দ পর্যালোচনা করুন।",
        "hospital_desc": "হাসপাতালের শয্যার চাহিদা, সক্ষমতা, ব্যবহার এবং একই শহরের মধ্যে স্থানান্তর পর্যালোচনা করুন।",
        "vector_desc": "পরিকল্পিত ভেক্টর নিয়ন্ত্রণ কার্যক্রম এবং সম্ভাব্য পরবর্তী প্রভাব দেখুন।",
        "dispatch_desc": "মাঠকর্মী ও স্থানীয় কর্মকর্তাদের জন্য ওয়ার্ডভিত্তিক কার্যক্রমের নোটিশ প্রস্তুত করুন।",
        "prototype_notice": "শুধু ফ্রন্টএন্ড প্রিভিউ। ব্যাকএন্ড গণনা এখনো যুক্ত হয়নি; কোনো কার্যকরী ফলাফল তৈরি হচ্ছে না।",
        "regional_stock": "আঞ্চলিকভাবে উপলব্ধ মজুত",
        "spraying_crews": "মোট উপলব্ধ স্প্রে দল",
        "stock_preview": "মজুতের প্রিভিউ",
        "crew_preview": "দলের সক্ষমতার প্রিভিউ",
        "saline": "স্যালাইন ব্যাগ",
        "diagnostic_kits": "ডায়াগনস্টিক কিট",
        "beds_required": "প্রয়োজনীয় হাসপাতাল শয্যা",
        "spraying_assignments": "স্প্রে দলের বরাদ্দ",
        "awaiting_backend": "ব্যাকএন্ড সংযোগের অপেক্ষায়",
        "hospital_capacity": "হাসপাতালের সক্ষমতার সারসংক্ষেপ",
        "bed_demand": "শয্যার চাহিদা",
        "available_beds": "উপলব্ধ শয্যা",
        "city_utilization": "শহরব্যাপী শয্যার ব্যবহার",
        "ward_deficits": "ওয়ার্ডভিত্তিক শয্যার ঘাটতি",
        "overflow_routes": "অতিরিক্ত রোগী স্থানান্তরের রুট",
        "select_wards": "কার্যক্রমের জন্য ওয়ার্ড নির্বাচন করুন",
        "intervention_teams": "উপলব্ধ ভেক্টর নিয়ন্ত্রণ দল",
        "simulation_horizon": "সিমুলেশনের সময়সীমা",
        "baseline_cases": "ভিত্তিমূলক আক্রান্ত",
        "simulated_cases": "সিমুলেশনের আক্রান্ত",
        "cases_prevented": "আনুমানিক প্রতিরোধকৃত আক্রান্ত",
        "dispatch_ward": "ওয়ার্ড নির্বাচন করুন",
        "dispatch_format": "ডিসপ্যাচের ফরম্যাট",
        "dispatch_preview": "ডিসপ্যাচ নথির প্রিভিউ",
        "generate_dispatch": "ডিসপ্যাচ নথি তৈরি করুন",
        "dispatch_unavailable": "এক্সপোর্ট মডিউল যুক্ত হলে ডিসপ্যাচ তৈরি চালু হবে।",
        "no_backend_results": "এখনো কোনো গণনাকৃত ফলাফল পাওয়া যায়নি।",
        "ui_control_note": "এটি ফ্রন্টএন্ড প্রিভিউ; এখনো সম্পদ বরাদ্দ পরিবর্তন করে না।",
        "select_channel": "আউটপুট ফরম্যাট নির্বাচন করুন"
    }
}

t = T[language]

def style_data_table(dataframe, risk_column):
    header_bg = "#0F766E"
    row_bg = "#14243A" if dark else "#FFFFFF"
    alternate_bg = "#1A2D45" if dark else "#F1F7FA"
    body_text = "#E8EEF7" if dark else "#24364B"
    grid_color = "#2B405A" if dark else "#DCE7EE"

    styled = dataframe.style.set_table_styles(
        [
            {"selector": "th", "props": [("background-color", header_bg), ("color", "#FFFFFF"), ("font-weight", "800"), ("font-size", "12px"), ("letter-spacing", "0.2px"), ("padding", "13px 14px"), ("border", f"1px solid {header_bg}")]},
            {"selector": "td", "props": [("color", body_text), ("font-size", "13px"), ("font-weight", "550"), ("padding", "12px 14px"), ("border-bottom", f"1px solid {grid_color}"), ("border-right", f"1px solid {grid_color}")]},
            {"selector": "tbody tr:nth-child(odd) td", "props": [("background-color", row_bg)]},
            {"selector": "tbody tr:nth-child(even) td", "props": [("background-color", alternate_bg)]},
            {"selector": "tbody tr:hover td", "props": [("background-color", "#174E55" if dark else "#DDF5F0"), ("color", "#FFFFFF" if dark else "#123B3A")]},
            {"selector": "table", "props": [("border-collapse", "separate"), ("border-spacing", "0"), ("border-radius", "10px"), ("overflow", "hidden")]},
        ]
    )

    if risk_column in dataframe.columns:
        def risk_style(value):
            risk = str(value).lower()
            palette = {
                "low": ("#14532D", "#BBF7D0"),
                "medium": ("#713F12", "#FDE68A"),
                "high": ("#7C2D12", "#FED7AA"),
                "critical": ("#7F1D1D", "#FECACA"),
                "কম ঝুঁকি": ("#14532D", "#BBF7D0"),
                "মাঝারি ঝুঁকি": ("#713F12", "#FDE68A"),
                "উচ্চ ঝুঁকি": ("#7C2D12", "#FED7AA"),
                "অতি উচ্চ ঝুঁকি": ("#7F1D1D", "#FECACA"),
            }
            background, foreground = palette.get(risk, ("#334155", "#F8FAFC"))
            return f"background-color: {background}; color: {foreground}; font-weight: 800; text-align: center; border-radius: 7px"

        styled = styled.map(risk_style, subset=[risk_column])

    return styled

if dark:
    bg = "#0B1220"
    panel = "#111C2E"
    card = "#162338"
    border = "#26364D"
    text = "#E8EEF7"
    muted = "#91A3B8"
    accent = "#2DD4BF"
    chart_template = "plotly_dark"
else:
    bg = "#F3F6FA"
    panel = "#FFFFFF"
    card = "#FFFFFF"
    border = "#DFE6EF"
    text = "#172B4D"
    muted = "#64748B"
    accent = "#0D9488"
    chart_template = "plotly_white"

st.markdown(
    f"""
    <style>
    #MainMenu {{
        display: none !important;
    }}

    .stApp {{
        background:
            radial-gradient(ellipse at 5% 0%, rgba(45, 212, 191, 0.10), transparent 35%),
            radial-gradient(ellipse at 95% 10%, rgba(99, 102, 241, 0.12), transparent 35%),
            {bg};
        color: {text};
    }}

    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {panel}, {bg});
        border-right: 1px solid {border};
    }}

    .sidebar-brand {{
        background: linear-gradient(135deg, rgba(45, 212, 191, 0.16), rgba(99, 102, 241, 0.12));
        border: 1px solid {border};
        border-radius: 18px;
        padding: 18px 15px;
        margin: 4px 0 18px 0;
    }}

    .sidebar-brand-icon {{
        width: 42px;
        height: 42px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(45, 212, 191, 0.16);
        border: 1px solid rgba(45, 212, 191, 0.35);
        border-radius: 13px;
        font-size: 23px;
        margin-bottom: 13px;
    }}

    .sidebar-brand-title {{
        color: {text};
        font-size: 20px;
        font-weight: 850;
        letter-spacing: -0.6px;
    }}

    .sidebar-brand-subtitle {{
        color: {muted};
        font-size: 10px;
        font-weight: 750;
        letter-spacing: 1.5px;
        margin-top: 5px;
    }}

    [data-testid="stSidebar"] [data-testid="stRadio"] label {{
        border-radius: 10px;
        padding: 7px 10px;
        transition: background 0.2s ease;
        font-weight: 600;
    }}

    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{
        background: rgba(45, 212, 191, 0.10);
    }}

    [data-testid="stHeader"] {{
        background: transparent;
    }}

    [data-testid="stMetric"] {{
        background:
            radial-gradient(circle at 100% 0%, rgba(45, 212, 191, 0.12), transparent 42%),
            linear-gradient(145deg, {card}, {panel});
        border: 1px solid {border};
        border-left: 4px solid {accent};
        padding: 21px 22px;
        border-radius: 18px;
        box-shadow: 0 8px 28px rgba(0, 0, 0, 0.12);
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
        min-height: 118px;
    }}

    [data-testid="stMetric"]:hover {{
        transform: translateY(-4px);
        border-color: {accent};
        box-shadow: 0 12px 32px rgba(45, 212, 191, 0.14);
    }}

    [data-testid="stMetricLabel"] {{
        color: {muted};
        font-weight: 600;
    }}

    [data-testid="stMetricValue"] {{
        color: {text};
        font-weight: 750;
    }}

    .risk-card {{
        background: linear-gradient(145deg, {card}, {panel});
        border: 1px solid {border};
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.10);
        transition: all 0.2s ease;
    }}

    .risk-card:hover {{
        transform: translateY(-4px);
    }}

    .risk-card-label {{
        font-size: 14px;
        font-weight: 650;
        margin-bottom: 12px;
    }}

    .risk-card-value {{
        font-size: 30px;
        font-weight: 800;
        color: {text};
    }}

    h1, h2, h3 {{
        color: {text};
        letter-spacing: -0.5px;
        font-weight: 700;
    }}

    p, label, .stCaption {{
        color: {muted};
    }}

    div[data-testid="stSelectbox"] > div > div {{
        border-radius: 11px;
        border-color: {border};
    }}

    div[data-testid="stButton"] button {{
        border-radius: 11px;
        min-height: 46px;
        font-weight: 700;
        transition: all 0.2s ease;
    }}

    div[data-testid="stButton"] button[kind="primary"] {{
        background: #0F766E !important;
        border: 1px solid #14B8A6 !important;
        color: #FFFFFF !important;
        font-weight: 800 !important;
        box-shadow: 0 4px 18px rgba(13, 148, 136, 0.25);
    }}

    div[data-testid="stButton"] button[kind="primary"]:hover {{
        background: #115E59 !important;
        border-color: #2DD4BF !important;
        color: #FFFFFF !important;
    }}

    div[data-testid="stButton"] button[kind="primary"] p {{
        color: #FFFFFF !important;
    }}

    [data-testid="stDataFrame"] {{
        border: 1px solid {border};
        border-radius: 14px;
        overflow: hidden;
    }}

    .hero {{
        background:
            radial-gradient(ellipse at 88% 12%, rgba(99, 102, 241, 0.22), transparent 35%),
            radial-gradient(ellipse at 12% 100%, rgba(45, 212, 191, 0.20), transparent 48%),
            linear-gradient(125deg, {panel}, {card});
        border: 1px solid {border};
        border-radius: 24px;
        padding: 38px 40px;
        margin: 8px 0 25px 0;
        box-shadow: 0 16px 42px rgba(0, 0, 0, 0.15);
        position: relative;
        overflow: hidden;
    }}

    .hero::after {{
        content: "";
        position: absolute;
        width: 245px;
        height: 245px;
        right: -75px;
        top: -105px;
        background: rgba(45, 212, 191, 0.18);
        filter: blur(58px);
        border-radius: 50%;
        pointer-events: none;
    }}

    .hero-eyebrow {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        color: #5EEAD4;
        background: rgba(13, 148, 136, 0.14);
        border: 1px solid rgba(45, 212, 191, 0.28);
        border-radius: 999px;
        padding: 6px 11px;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 1.6px;
        text-transform: uppercase;
        margin-bottom: 13px;
    }}

    .hero-title {{
        color: {text};
        font-size: clamp(30px, 4vw, 43px);
        font-weight: 850;
        letter-spacing: -1.5px;
        line-height: 1.12;
        margin: 0 0 12px 0;
        position: relative;
        z-index: 1;
    }}

    .hero-description {{
        color: {muted};
        font-size: 15px;
        margin-bottom: 0;
        max-width: 720px;
        line-height: 1.8;
        position: relative;
        z-index: 1;
    }}

    [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] {{
        gap: 1rem;
    }}

    [data-testid="stSubheader"] {{
        padding-top: 5px;
    }}

    .risk-note {{
        background: linear-gradient(110deg, {card}, {panel});
        border: 1px solid {border};
        border-left: 3px solid #F59E0B;
        padding: 15px 18px;
        border-radius: 12px;
        color: {muted};
        font-size: 12px;
    }}

    .section-heading {{
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 17px 20px;
        margin: 18px 0 13px 0;
        border: 1px solid {border};
        border-radius: 17px;
        background:
            radial-gradient(circle at 100% 0%, rgba(99, 102, 241, 0.16), transparent 38%),
            linear-gradient(115deg, rgba(45, 212, 191, 0.11), {card} 58%, {panel});
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.10);
    }}

    .section-heading-icon {{
        width: 43px;
        height: 43px;
        min-width: 43px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 13px;
        background: rgba(45, 212, 191, 0.14);
        border: 1px solid rgba(45, 212, 191, 0.30);
        font-size: 21px;
    }}

    .section-heading-title {{
        color: {text};
        font-size: 18px;
        line-height: 1.25;
        font-weight: 800;
        letter-spacing: -0.35px;
        margin: 0;
    }}

    .section-heading-subtitle {{
        color: {muted};
        font-size: 11px;
        line-height: 1.5;
        margin-top: 4px;
    }}

    .section-table-wrap {{
        padding: 5px 8px 8px 8px;
        border: 1px solid {border};
        border-radius: 17px;
        background: linear-gradient(145deg, {panel}, {card});
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
    }}

    .coverage-strip {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 14px;
        padding: 13px 16px;
        margin: 2px 0 12px 0;
        border: 1px solid rgba(45, 212, 191, 0.28);
        border-radius: 14px;
        background: linear-gradient(105deg, rgba(13, 148, 136, 0.13), {card});
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.08);
    }}

    .coverage-label {{
        color: {muted};
        font-size: 12px;
        font-weight: 650;
    }}

    .coverage-value {{
        color: {text};
        font-size: 13px;
        font-weight: 850;
        white-space: nowrap;
    }}

    .coverage-value span {{
        display: inline-block;
        color: #5EEAD4;
        background: rgba(13, 148, 136, 0.18);
        border: 1px solid rgba(45, 212, 191, 0.30);
        border-radius: 999px;
        padding: 5px 10px;
        margin-left: 7px;
    }}

    .table-intro {{
        display: flex;
        align-items: center;
        gap: 9px;
        color: {muted};
        font-size: 11px;
        font-weight: 650;
        padding: 9px 12px;
        margin: 0 0 10px 0;
        border-radius: 10px;
        background: rgba(45, 212, 191, 0.07);
        border: 1px solid {border};
    }}

    .risk-guide {{
        border: 1px solid {border};
        border-radius: 21px;
        padding: 22px;
        margin: 20px 0;
        background:
            radial-gradient(circle at 100% 0%, rgba(99, 102, 241, 0.13), transparent 34%),
            linear-gradient(135deg, {panel}, {card});
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.10);
    }}

    .risk-guide-title {{
        color: {text};
        font-size: 20px;
        font-weight: 850;
        letter-spacing: -0.5px;
        margin-bottom: 5px;
    }}

    .risk-guide-description {{
        color: {muted};
        font-size: 12px;
        line-height: 1.7;
        margin-bottom: 17px;
    }}

    .risk-threshold-grid {{
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
    }}

    .risk-threshold-card {{
        border: 1px solid {border};
        border-top: 4px solid var(--risk-color);
        border-radius: 15px;
        padding: 16px 14px;
        background: {card};
        min-height: 126px;
    }}

    .risk-threshold-name {{
        color: var(--risk-color);
        font-size: 13px;
        font-weight: 850;
        margin-bottom: 10px;
    }}

    .risk-threshold-value {{
        color: {text};
        font-size: 22px;
        font-weight: 850;
        letter-spacing: -0.5px;
        margin-bottom: 6px;
    }}

    .risk-threshold-description {{
        color: {muted};
        font-size: 11px;
        line-height: 1.55;
    }}

    .risk-guide-method {{
        color: {muted};
        font-size: 15px;
        line-height: 1.75;
        padding: 12px 14px;
        margin-top: 14px;
        border-radius: 12px;
        background: rgba(45, 212, 191, 0.07);
        border: 1px solid {border};
    }}

    @media (max-width: 700px) {{
        .risk-threshold-grid {{
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }}
    }}

    [data-testid="stVerticalBlockBorderWrapper"] {{
        border: 1px solid {border} !important;
        border-radius: 17px !important;
        background: linear-gradient(145deg, {panel}, {card}) !important;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
        padding: 8px !important;
    }}

    [data-testid="stDataFrame"] {{
        border: 1px solid {border};
        border-radius: 13px;
        overflow: hidden;
        background: {panel};
    }}

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {{
        color: {text};
    }}

    hr {{
        border-color: {border};
        opacity: 0.7;
    }}

    footer {{
        visibility: hidden;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    return get_clean_data()


@st.cache_data
def load_all_ward_forecasts(clean_data, as_of_date):
    return predict_all_wards(clean_data, as_of_date=as_of_date)


def render_risk_guide():
    risk_cards = [
        ("LOW RISK" if language == "English" else "কম ঝুঁকি", "< 3", t["risk_guide_low"], "#22C55E"),
        ("MEDIUM RISK" if language == "English" else "মাঝারি ঝুঁকি", "3 – < 6", t["risk_guide_medium"], "#EAB308"),
        ("HIGH RISK" if language == "English" else "উচ্চ ঝুঁকি", "6 – < 10", t["risk_guide_high"], "#F97316"),
        ("CRITICAL" if language == "English" else "অতি উচ্চ ঝুঁকি", "≥ 10", t["risk_guide_critical"], "#EF4444"),
    ]

    cards_html = "".join(
        f"""
        <div class="risk-threshold-card" style="--risk-color: {color};">
            <div class="risk-threshold-name">{name}</div>
            <div class="risk-threshold-value">{threshold}</div>
            <div class="risk-threshold-description">{description}</div>
        </div>
        """
        for name, threshold, description, color in risk_cards
    )

    st.markdown(
        f"""
        <div class="risk-guide">
            <div class="risk-guide-title"> {t['risk_guide_title']}</div>
            <div class="risk-guide-description">{t['risk_guide_desc']}</div>
            <div class="risk-threshold-grid">{cards_html}</div>
            <div class="risk-guide-method">
                <strong>{t['risk_guide_unit']}</strong><br>
                {t['risk_guide_method']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-icon">🦟</div>
            <div class="sidebar-brand-title">DengueGuard AI</div>
            <div class="sidebar-brand-subtitle">PUBLIC HEALTH INTELLIGENCE</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f"**{t['navigation']}**")

    page_labels = {
        "dashboard": t["dashboard"],
        "ward_forecast": t["ward_forecast"],
        "resource_allocation": t["resource_allocation"],
        "hospital_surge": t["hospital_surge"],
        "vector_simulator": t["vector_simulator"],
        "dispatch_center": t["dispatch_center"],
    }

    page = st.radio(
        "Navigation",
        list(page_labels.keys()),
        format_func=lambda page_id: page_labels[page_id],
        label_visibility="collapsed",
        key="page_navigation",
    )

    st.divider()
    st.markdown(f"**{t['system']}**")

    st.selectbox(
        t["theme"],
        ["Dark", "Light"],
        index=0 if dark else 1,
        key="theme",
    )

    st.selectbox(
        t["language"],
        ["English", "বাংলা"],
        index=0 if language == "English" else 1,
        key="language",
    )

    st.divider()
    st.markdown(f"**{t['data_period']}**")
    st.caption("2016 — 2025")
    st.caption("60 wards · 5 urban areas")
    st.caption("2, 3 and 4-week forecasts")

try:
    with st.spinner(t["loading"]):
        clean_data = load_data()
except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()

available_dates = pd.to_datetime(clean_data[DATE_COL]).dropna()
available_dates = available_dates.sort_values().drop_duplicates()

min_date = available_dates.min()
max_date = available_dates.max()

with st.sidebar:
    st.divider()
    st.markdown(f"**{t['date']}**")

    selected_day = st.selectbox(
        t["day"],
        list(range(1, 32)),
        index=28,
    )

    month_names = [
        "January", "February", "March", "April",
        "May", "June", "July", "August",
        "September", "October", "November", "December",
    ]

    bangla_months = [
        "জানুয়ারি", "ফেব্রুয়ারি", "মার্চ", "এপ্রিল",
        "মে", "জুন", "জুলাই", "আগস্ট",
        "সেপ্টেম্বর", "অক্টোবর", "নভেম্বর", "ডিসেম্বর",
    ]

    selected_month = st.selectbox(
        t["month"],
        range(1, 13),
        index=11,
        format_func=lambda month: (
            month_names[month - 1]
            if language == "English"
            else bangla_months[month - 1]
        ),
    )

    selected_year = st.selectbox(
        t["year"],
        list(range(min_date.year, max_date.year + 1)),
        index=max_date.year - min_date.year,
    )

try:
    requested_date = pd.Timestamp(
        year=selected_year,
        month=selected_month,
        day=selected_day,
    )

    matching_dates = available_dates[
        available_dates <= requested_date
    ]

    if requested_date < min_date or matching_dates.empty:
        st.error(t["no_date"])
        st.stop()

    selected_as_of_date = matching_dates.max()

except ValueError:
    st.error(t["invalid_date"])
    st.stop()

as_of_date = selected_as_of_date.strftime("%Y-%m-%d")

st.markdown(
    f"""
    <div class="hero">
        <div class="hero-eyebrow">● &nbsp; EARLY WARNING INTELLIGENCE</div>
        <div class="hero-title">🦟 DengueGuard AI</div>
        <p class="hero-description">{t['intro']}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    f"{t['selected_date']}: {requested_date.strftime('%d %B %Y')}  ·  "
    f"{t['available_date']}: {selected_as_of_date.strftime('%d %B %Y')}"
)

if page == "dashboard":
    st.markdown(f"### {t['overview']}")
    st.caption(t["overview_desc"])

    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        st.metric(t["records"], f"{len(clean_data):,}")

    with col2:
        st.metric(t["wards"], clean_data[WARD_COL].nunique())

    with col3:
        st.metric(
            t["latest"],
            max_date.strftime("%d %b %Y"),
        )

    st.divider()

    try:
        with st.spinner(t["generating"]):
            all_forecasts = load_all_ward_forecasts(
                clean_data,
                as_of_date,
            )

        filter_col1, filter_col2 = st.columns(2)

        with filter_col1:
            cities = ["All Bangladesh"] + sorted(
                all_forecasts[CITY_COL].dropna().unique().tolist()
            )

            selected_city = st.selectbox(
                t["city"],
                cities,
                format_func=lambda city: (
                    t["all"] if city == "All Bangladesh" else city
                ),
            )

        with filter_col2:
            selected_horizon = st.selectbox(
                t["horizon"],
                [2, 3, 4],
                format_func=lambda x: f"{x} Weeks Ahead",
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

        risk_labels = [
            t["low"],
            t["medium"],
            t["high"],
            t["critical"],
        ]

        metric_cols = st.columns(4)

        risk_colors = {
            "Low": "#22C55E",
            "Medium": "#EAB308",
            "High": "#F97316",
            "Critical": "#EF4444",
        }

        for col, risk, label in zip(
            metric_cols,
            risk_order,
            risk_labels,
        ):
            with col:
                color = risk_colors[risk]

                st.markdown(
                    f"""
                    <div class="risk-card" style="border-top: 4px solid {color};">
                        <div class="risk-card-label" style="color: {color};">
                            {label}
                        </div>
                        <div class="risk-card-value">
                            {int(risk_counts[risk])}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.divider()
        st.markdown(f"### {t['distribution']}")

        risk_chart_data = pd.DataFrame({
            "Risk Category": risk_labels,
            "Number of Wards": [
                int(risk_counts[risk]) for risk in risk_order
            ],
        })

        fig = px.bar(
            risk_chart_data,
            x="Risk Category",
            y="Number of Wards",
            color="Risk Category",
            category_orders={"Risk Category": risk_labels},
            text="Number of Wards",
            color_discrete_map={
                risk_labels[0]: "#22C55E",
                risk_labels[1]: "#EAB308",
                risk_labels[2]: "#F97316",
                risk_labels[3]: "#EF4444",
            },
            template=chart_template,
        )

        fig.update_layout(
            showlegend=False,
            yaxis_title="Number of Wards",
            xaxis_title="",
            yaxis_rangemode="tozero",
            margin=dict(l=10, r=10, t=25, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        fig.update_traces(
            textposition="outside",
            cliponaxis=False,
        )

        st.plotly_chart(fig, width='stretch')

        st.markdown(
            f"""
            <div class="section-heading">
                <div class="section-heading-icon">🗺️</div>
                <div>
                    <div class="section-heading-title">{t['details']}</div>
                    <div class="section-heading-subtitle">{t["ward_detail_desc"]}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        overview_table = overview[
            [
                WARD_COL,
                CITY_COL,
                "predicted_cases",
                "risk_category",
            ]
        ].copy()

        overview_table.columns = [
            t["ward"],
            t["city_name"],
            t["predicted"],
            t["risk"],
        ]

        overview_table = overview_table.sort_values(
            t["ward"],
            ascending=True,
        ).reset_index(drop=True)

        st.markdown(
            f"""
            <div class="coverage-strip">
                <div class="coverage-label">🌍 {t['wards']}</div>
                <div class="coverage-value">{t['showing']} <span>{len(overview_table)} / {all_forecasts[WARD_COL].nunique()}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            st.markdown(
                f'<div class="table-intro"> {t["ward_detail_desc"]}</div>',
                unsafe_allow_html=True,
            )
            st.dataframe(
                style_data_table(overview_table, t["risk"]),
                width='stretch',
                height=550,
                hide_index=True,
                column_config={
                    t["predicted"]: st.column_config.NumberColumn(
                        format="%.2f"
                    )
                },
            )

        st.markdown(
            f'<div class="risk-note">{t["risk_note"]}</div>',
            unsafe_allow_html=True,
        )

    except Exception as exc:
        st.error(f"Could not generate the all-ward overview: {exc}")

elif page == "ward_forecast":
    st.markdown(f"### {t['ward_forecast']}")
    st.caption(t["forecast_page_desc"])

    wards = sorted(clean_data[WARD_COL].dropna().unique())

    selected_ward = st.selectbox(
        t["select_ward"],
        wards,
    )

    if st.button(t["generate"], type="primary", width='stretch'):
        try:
            with st.spinner(t["generating"]):
                forecasts = predict_ward(
                    clean_data=clean_data,
                    ward_id=selected_ward,
                    as_of_date=as_of_date,
                )

            st.success(t["success"])

            city_name = (
                forecasts[CITY_COL].iloc[0]
                if CITY_COL in forecasts.columns
                else "Unknown"
            )

            forecast_date = pd.to_datetime(
                forecasts[DATE_COL].iloc[0]
            ).strftime("%d %b %Y")

            st.caption(
                f"{t['ward']}: {selected_ward}  ·  "
                f"{t['city_name']}: {city_name}  ·  "
                f"{t['issued']}: {forecast_date}"
            )

            st.divider()
            st.markdown(f"### {t['drivers']}")

            driver_date = pd.to_datetime(
                forecasts[DATE_COL].iloc[0]
            )

            driver_row = clean_data[
                (clean_data[WARD_COL] == selected_ward)
                & (
                    pd.to_datetime(clean_data[DATE_COL])
                    == driver_date
                )
            ]

            if not driver_row.empty:
                drivers = driver_row.iloc[0]

                driver_cols = st.columns(3)

                with driver_cols[0]:
                    st.metric(
                        t["temperature"],
                        f"{drivers['avg_temp_c']:.1f} °C",
                    )
                    st.metric(
                        t["rainfall"],
                        f"{drivers['rainfall_mm']:.1f} mm",
                    )

                with driver_cols[1]:
                    st.metric(
                        t["humidity"],
                        f"{drivers['relative_humidity_pct']:.1f}%",
                    )
                    st.metric(
                        t["waterlogging"],
                        f"{drivers['water_logging_risk_index']:.2f}",
                    )

                with driver_cols[2]:
                    st.metric(
                        t["density"],
                        f"{drivers['population_density_sqkm']:,.1f} / km²",
                    )
                    st.metric(
                        t["sanitation"],
                        f"{drivers['sanitation_score']:.2f}",
                    )

            else:
                st.warning(t["unavailable"])

            st.divider()
            st.markdown(f"### {t['results']}")

            forecast_cols = st.columns(3)

            for col, (_, row) in zip(
                forecast_cols,
                forecasts.iterrows(),
            ):
                with col:
                    st.metric(
                        f"{row['horizon_weeks']}-Week Forecast",
                        f"{row['predicted_cases']:.2f}",
                    )

                    risk = row["risk_category"]

                    if risk == "Low":
                        st.success(f"{t['risk']}: {t['low']}")
                    elif risk == "Medium":
                        st.warning(f"{t['risk']}: {t['medium']}")
                    else:
                        st.error(
                            f"{t['risk']}: "
                            f"{t[risk.lower()]}"
                        )

            st.markdown(
                f"""
                <div class="section-heading">
                    <div class="section-heading-icon">📋</div>
                    <div>
                        <div class="section-heading-title">{t['forecast_details']}</div>
                        <div class="section-heading-subtitle">{t["forecast_detail_desc"]}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            display_df = forecasts[
                [
                    "horizon_weeks",
                    "predicted_cases",
                    "risk_category",
                ]
            ].copy()

            display_df.columns = [
                t["horizon_weeks"],
                t["predicted"],
                t["risk"],
            ]

            with st.container(border=True):
                st.markdown(
                    f'<div class="table-intro"> {t["forecast_detail_desc"]}</div>',
                    unsafe_allow_html=True,
                )
                st.dataframe(
                    style_data_table(display_df, t["risk"]),
                    width='stretch',
                    hide_index=True,
                    column_config={
                        t["predicted"]: st.column_config.NumberColumn(
                            format="%.2f"
                        )
                    },
                )

            st.markdown(f"### {t['trend']}")

            chart_data = forecasts.copy()
            chart_data["Forecast Horizon"] = (
                chart_data["horizon_weeks"].astype(str) + " Weeks"
            )

            fig = px.bar(
                chart_data,
                x="Forecast Horizon",
                y="predicted_cases",
                text="predicted_cases",
                template=chart_template,
                color="risk_category",
                color_discrete_map={
                    "Low": "#22C55E",
                    "Medium": "#EAB308",
                    "High": "#F97316",
                    "Critical": "#EF4444",
                },
                labels={
                    "predicted_cases": t["cases"],
                    "Forecast Horizon": t["period"],
                    "risk_category": t["risk"],
                },
            )

            fig.update_traces(
                texttemplate="%{text:.2f}",
                textposition="outside",
            )

            fig.update_layout(
                showlegend=False,
                yaxis_rangemode="tozero",
                margin=dict(l=10, r=10, t=25, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )

            st.plotly_chart(fig, width='stretch')

        except Exception as exc:
            st.error(f"Forecast generation failed: {exc}")


elif page == "resource_allocation":
    st.markdown(f"### {t['resource_allocation']}")
    st.caption(t["resource_desc"])

    st.info(
        "Inventory and demand coefficients must be supplied by the "
        "operator. Historical resource requirements are not treated "
        "as available stock."
    )

    cities = sorted(clean_data[CITY_COL].dropna().unique().tolist())

    with st.form("resource_allocation_form"):
        st.markdown("#### Forecast configuration")

        config_col1, config_col2 = st.columns(2)

        with config_col1:
            selected_horizon = st.selectbox(
                "Forecast horizon",
                [2, 3, 4],
                format_func=lambda value: f"{value} weeks ahead",
            )

        with config_col2:
            stock_percentage = st.slider(
                "Available stock percentage",
                min_value=30,
                max_value=100,
                value=100,
                step=5,
            )

        st.divider()
        st.markdown("#### Regional inventory")

        inventory_records = []

        for city in cities:
            st.markdown(f"**{city}**")

            inventory_col1, inventory_col2 = st.columns(2)

            with inventory_col1:
                saline_stock = st.number_input(
                    "Saline bags available",
                    min_value=0,
                    value=0,
                    step=1,
                    key=f"saline_stock_{city}",
                )

            with inventory_col2:
                diagnostic_stock = st.number_input(
                    "Diagnostic kits available",
                    min_value=0,
                    value=0,
                    step=1,
                    key=f"diagnostic_stock_{city}",
                )

            inventory_records.append({
                "city": city,
                "saline_bags": saline_stock,
                "diagnostic_kits": diagnostic_stock,
            })

        st.divider()
        st.markdown("#### Demand assumptions")

        st.caption(
            "Enter estimated resource requirements per predicted dengue "
            "case and provide the source of each assumption."
        )

        demand_col1, demand_col2 = st.columns(2)

        with demand_col1:
            saline_per_case = st.number_input(
                "Saline bags per case",
                min_value=0.0,
                value=0.0,
                step=0.1,
                format="%.3f",
            )

            saline_source = st.text_input(
                "Saline demand source",
                placeholder="Enter guideline, study, or reference",
            )

        with demand_col2:
            diagnostic_per_case = st.number_input(
                "Diagnostic kits per case",
                min_value=0.0,
                value=0.0,
                step=0.1,
                format="%.3f",
            )

            diagnostic_source = st.text_input(
                "Diagnostic demand source",
                placeholder="Enter guideline, study, or reference",
            )

        submitted = st.form_submit_button(
            "Generate Resource Allocation",
            type="primary",
            width="stretch",
        )

    if submitted:
        if saline_per_case <= 0 or diagnostic_per_case <= 0:
            st.error(
                "Both per-case demand assumptions must be greater than zero."
            )

        elif not saline_source.strip() or not diagnostic_source.strip():
            st.error(
                "Please provide a source for both demand assumptions."
            )

        else:
            try:
                with st.spinner(
                    "Generating forecasts and allocating resources..."
                ):
                    forecasts = load_all_ward_forecasts(
                        clean_data,
                        as_of_date,
                    )

                    ward_information = (
                        clean_data[
                            [
                                WARD_COL,
                                CITY_COL,
                                "population_density_sqkm",
                            ]
                        ]
                        .drop_duplicates(subset=[WARD_COL])
                        .rename(columns={
                            WARD_COL: "ward_id",
                            CITY_COL: "city",
                        })
                    )

                    regional_inventory = pd.DataFrame(inventory_records)

                    demand_parameters = {
                        "saline_bags": {
                            "per_case": saline_per_case,
                            "source": saline_source.strip(),
                            "status": "user_supplied",
                        },
                        "diagnostic_kits": {
                            "per_case": diagnostic_per_case,
                            "source": diagnostic_source.strip(),
                            "status": "user_supplied",
                        },
                    }

                    result = allocate_resources(
                        forecasts=forecasts,
                        ward_information=ward_information,
                        regional_inventory=regional_inventory,
                        stock_percentage=stock_percentage,
                        demand_parameters=demand_parameters,
                        horizon_weeks=selected_horizon,
                    )

                    st.session_state["resource_allocation_result"] = result
                    st.session_state["resource_allocation_as_of_date"] = as_of_date

                    st.success(
                        "Resource allocation generated successfully."
                    )

            except Exception as exc:
                st.error(f"Resource allocation failed: {exc}")

    result = st.session_state.get("resource_allocation_result")
    result_date = st.session_state.get("resource_allocation_as_of_date")

    if result is not None and result_date == as_of_date:
        st.divider()
        st.markdown("#### Nationwide allocation summary")

        st.caption(
            f"Forecast date: {as_of_date} | "
            f"Horizon: {result['horizon_weeks']} weeks | "
            f"Stock availability: {result['stock_percentage']}%"
        )

        resource_summary = result["resource_summary"]
        metric_cols = st.columns(4)

        for index, resource in enumerate(
            ["saline_bags", "diagnostic_kits"]
        ):
            resource_row = resource_summary[
                resource_summary["resource"] == resource
            ]

            if resource_row.empty:
                continue

            row = resource_row.iloc[0]

            if index == 0:
                columns = metric_cols[:2]
                label = "Saline bags"
            else:
                columns = metric_cols[2:]
                label = "Diagnostic kits"

            with columns[0]:
                st.metric(
                    f"{label} allocated",
                    f"{int(row['total_allocated']):,}",
                )

            with columns[1]:
                st.metric(
                    f"{label} unmet",
                    f"{int(row['total_unmet']):,}",
                )

        st.markdown("#### Nationwide resource summary")

        st.dataframe(
            resource_summary,
            width="stretch",
            hide_index=True,
        )

        st.markdown("#### City-level allocation")

        st.dataframe(
            result["regional_summary"],
            width="stretch",
            hide_index=True,
        )

        st.markdown("#### Ward-level allocation")

        ward_results = result["ward_allocations"].copy()

        selected_risk = st.multiselect(
            "Filter by risk category",
            ["Critical", "High", "Medium", "Low"],
            default=["Critical", "High", "Medium", "Low"],
            key="allocation_risk_filter",
        )

        ward_results = ward_results[
            ward_results["risk_category"].isin(selected_risk)
        ]

        st.dataframe(
            ward_results,
            width="stretch",
            hide_index=True,
        )

        st.download_button(
            "Download ward allocation CSV",
            data=ward_results.to_csv(index=False).encode("utf-8"),
            file_name="dengueguard_ward_allocation.csv",
            mime="text/csv",
        )

    elif result is not None:
        st.info(
            "The forecast date has changed. Submit the allocation "
            "form again to generate results for the new date."
        )
elif page == "hospital_surge":
    st.markdown(f"### {t['hospital_surge']}")
    st.caption(t["hospital_desc"])

    st.info(
        "Hospital capacity must represent actual available beds. "
        "Historical hospital_beds_required values are not capacity data."
    )

    forecast_horizon = st.selectbox(
        "Forecast horizon",
        [2, 3, 4],
        format_func=lambda value: f"{value} weeks ahead",
        key="hospital_surge_horizon",
    )

    beds_per_case = st.number_input(
        "Estimated beds required per predicted dengue case",
        min_value=0.001,
        max_value=2.0,
        value=float(DEFAULT_BEDS_PER_CASE),
        step=0.001,
        format="%.3f",
        help="Scenario assumption. The default is not a clinically validated ratio.",
    )

    near_capacity_threshold = st.slider(
        "Near-capacity warning threshold",
        min_value=50,
        max_value=100,
        value=80,
        step=5,
    ) / 100

    try:
        all_forecasts = load_all_ward_forecasts(
            clean_data,
            as_of_date,
        )

        forecast_data = all_forecasts[
            all_forecasts["horizon_weeks"] == forecast_horizon
        ].copy()

        forecast_data = forecast_data.rename(
            columns={
                WARD_COL: "ward_id",
                CITY_COL: "city",
            }
        )

        forecast_data["ward_id"] = (
            forecast_data["ward_id"].astype(str)
        )

        forecast_data["city"] = (
            forecast_data["city"].astype(str)
        )

        forecast_data = forecast_data[
            [
                "ward_id",
                "city",
                "horizon_weeks",
                "predicted_cases",
            ]
        ]

        if forecast_data.empty:
            st.warning("No forecasts are available for this horizon.")
            st.stop()

    except Exception as error:
        st.error(f"Could not load forecasts: {error}")
        st.stop()

    st.markdown("#### Hospital Capacity Configuration")

    st.caption(
        "Provide the actual available bed capacity for every forecast ward."
    )

    template = forecast_data[
        ["ward_id", "city"]
    ].drop_duplicates().copy()

    template["capacity_beds"] = ""

    st.download_button(
        "Download hospital capacity CSV template",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="hospital_capacity_template.csv",
        mime="text/csv",
    )

    input_method = st.radio(
        "Capacity input method",
        ["Manual entry", "Upload CSV"],
        horizontal=True,
        key="hospital_capacity_input_method",
    )

    capacity_data = None

    if input_method == "Manual entry":
        manual_capacity = template.copy()
        manual_capacity["capacity_beds"] = pd.Series(
            [pd.NA] * len(manual_capacity),
            dtype="Int64",
        )

        capacity_data = st.data_editor(
            manual_capacity,
            hide_index=True,
            width="stretch",
            num_rows="fixed",
            column_config={
                "ward_id": st.column_config.TextColumn(
                    "Ward ID",
                    disabled=True,
                ),
                "city": st.column_config.TextColumn(
                    "City",
                    disabled=True,
                ),
                "capacity_beds": st.column_config.NumberColumn(
                    "Available beds",
                    min_value=0,
                    step=1,
                    format="%d",
                ),
            },
            key="hospital_capacity_editor",
        )

    else:
        uploaded_capacity = st.file_uploader(
            "Upload hospital capacity CSV",
            type=["csv"],
            key="hospital_capacity_upload",
        )

        if uploaded_capacity is not None:
            try:
                capacity_data = pd.read_csv(uploaded_capacity)
            except Exception as error:
                st.error(f"Could not read CSV: {error}")

    if capacity_data is not None:

        required_columns = {
            "ward_id",
            "city",
            "capacity_beds",
        }

        missing_columns = (
            required_columns - set(capacity_data.columns)
        )

        if missing_columns:
            st.error(
                f"Missing CSV columns: {sorted(missing_columns)}"
            )

        else:
            capacity_data = capacity_data[
                ["ward_id", "city", "capacity_beds"]
            ].copy()

            capacity_data["ward_id"] = (
                capacity_data["ward_id"].astype(str)
            )

            capacity_data["city"] = (
                capacity_data["city"].astype(str)
            )

            numeric_capacity = pd.to_numeric(
                capacity_data["capacity_beds"],
                errors="coerce",
            )

            forecast_wards = set(forecast_data["ward_id"])
            capacity_wards = set(capacity_data["ward_id"])

            missing_wards = forecast_wards - capacity_wards

            valid_capacity = (
                numeric_capacity.notna().all()
                and numeric_capacity.ge(0).all()
                and numeric_capacity.mod(1).eq(0).all()
            )

            valid_wards = (
                not capacity_data["ward_id"].duplicated().any()
                and not missing_wards
                and capacity_data["ward_id"].notna().all()
                and capacity_data["city"].notna().all()
            )

            if not valid_wards:
                st.error(
                    "Capacity data contains duplicate wards, "
                    "missing wards, or invalid ward information."
                )

            elif not valid_capacity:
                st.error(
                    "Available bed capacity must contain "
                    "nonnegative whole numbers for every ward."
                )

            else:
                capacity_data["capacity_beds"] = (
                    numeric_capacity.astype(int)
                )

                try:
                    assessment = assess_hospital_surge(
                        forecast_data,
                        capacity_data,
                        beds_per_case=beds_per_case,
                        near_capacity_threshold=near_capacity_threshold,
                    )

                    overflow_plan = generate_overflow_plan(
                        assessment
                    )

                    st.success(
                        "Hospital surge assessment generated successfully."
                    )

                    total_demand = int(
                        assessment["estimated_bed_demand"].sum()
                    )

                    total_capacity = int(
                        assessment["capacity_beds"].sum()
                    )

                    total_deficit = int(
                        assessment["excess_beds"].sum()
                    )

                    total_headroom = int(
                        assessment["spare_beds"].sum()
                    )

                    overall_utilization = (
                        f"{total_demand / total_capacity:.1%}"
                        if total_capacity > 0
                        else (
                            "No capacity"
                            if total_demand > 0
                            else "0%"
                        )
                    )

                    st.markdown("#### Overall Hospital Situation")

                    metric_cols = st.columns(4)

                    metric_cols[0].metric(
                        "Estimated bed demand",
                        f"{total_demand:,}",
                    )

                    metric_cols[1].metric(
                        "Available beds",
                        f"{total_capacity:,}",
                    )

                    metric_cols[2].metric(
                        "Bed deficit",
                        f"{total_deficit:,}",
                    )

                    metric_cols[3].metric(
                        "Overall utilization",
                        overall_utilization,
                    )

                    city_summary = (
                        assessment.groupby("city")
                        .agg(
                            estimated_demand=(
                                "estimated_bed_demand",
                                "sum",
                            ),
                            available_beds=(
                                "capacity_beds",
                                "sum",
                            ),
                            headroom=(
                                "spare_beds",
                                "sum",
                            ),
                            bed_deficit=(
                                "excess_beds",
                                "sum",
                            ),
                        )
                        .reset_index()
                    )

                    city_summary["utilization"] = (
                        city_summary.apply(
                            lambda row: (
                                f"{row['estimated_demand'] / row['available_beds']:.1%}"
                                if row["available_beds"] > 0
                                else (
                                    "No capacity"
                                    if row["estimated_demand"] > 0
                                    else "0%"
                                )
                            ),
                            axis=1,
                        )
                    )

                    st.markdown("#### City-Wide Bed Utilization")

                    st.dataframe(
                        city_summary,
                        width="stretch",
                        hide_index=True,
                    )

                    ward_results = assessment.copy()

                    ward_results["utilization"] = (
                        ward_results["capacity_utilization"].apply(
                            lambda value: (
                                "No capacity"
                                if value == float("inf")
                                else f"{value:.1%}"
                            )
                        )
                    )

                    ward_results = ward_results.drop(
                        columns=["capacity_utilization"]
                    )

                    st.markdown("#### Ward-Level Hospital Assessment")

                    st.dataframe(
                        ward_results,
                        width="stretch",
                        hide_index=True,
                    )

                    deficits = assessment[
                        assessment["excess_beds"] > 0
                    ].copy()

                    st.markdown("#### Overloaded Wards")

                    if deficits.empty:
                        st.success(
                            "No wards currently exceed their available bed capacity."
                        )
                    else:
                        st.dataframe(
                            deficits[
                                [
                                    "ward_id",
                                    "city",
                                    "estimated_bed_demand",
                                    "capacity_beds",
                                    "excess_beds",
                                    "surge_status",
                                ]
                            ],
                            width="stretch",
                            hide_index=True,
                        )

                    st.markdown("#### Cross-Ward Overflow Routing")

                    if overflow_plan.empty:
                        st.info(
                            "No overflow transfers are currently required."
                        )
                    else:
                        display_plan = overflow_plan.rename(
                            columns={
                                "from_ward": "Ward needing beds",
                                "to_ward": "Donor ward",
                                "beds_to_transfer": "Beds recommended",
                                "unmet_beds": "Unmet beds",
                            }
                        )

                        st.dataframe(
                            display_plan,
                            width="stretch",
                            hide_index=True,
                        )

                        unmet_total = int(
                            overflow_plan["unmet_beds"].sum()
                        )

                        if unmet_total > 0:
                            st.error(
                                f"{unmet_total} beds remain unmet "
                                "after possible same-city routing."
                            )

                    st.download_button(
                        "Download hospital surge assessment",
                        data=ward_results.to_csv(
                            index=False
                        ).encode("utf-8"),
                        file_name="hospital_surge_assessment.csv",
                        mime="text/csv",
                    )

                    st.download_button(
                        "Download overflow routing plan",
                        data=overflow_plan.to_csv(
                            index=False
                        ).encode("utf-8"),
                        file_name="hospital_overflow_plan.csv",
                        mime="text/csv",
                    )

                    st.caption(
                        "Overflow routes are planning recommendations, "
                        "not confirmed patient transfers. Bed demand is "
                        "a scenario estimate."
                    )

                except (ValueError, TypeError) as error:
                    st.error(f"Hospital assessment error: {error}")

elif page == "vector_simulator":
    st.markdown(f"### {t['vector_simulator']}")
    st.caption(t["vector_desc"])

    available_wards = sorted(
        clean_data[WARD_COL].dropna().unique().tolist()
    )

    selected_wards = st.multiselect(
        t["select_wards"],
        available_wards,
        key="vector_ward_preview",
    )

    control_col, horizon_col = st.columns(2)

    with control_col:
        intervention_teams = st.number_input(
            t["intervention_teams"],
            min_value=0,
            max_value=1000,
            value=0,
            step=1,
            key="vector_team_preview",
        )

    with horizon_col:
        simulation_horizon = st.selectbox(
            t["simulation_horizon"],
            [2, 3, 4],
            format_func=lambda weeks: f"{weeks} weeks",
            key="vector_horizon_preview",
        )

    st.caption(t["ui_control_note"])

    metric_cols = st.columns(3)

    for col, label in zip(
        metric_cols,
        [
            t["baseline_cases"],
            t["simulated_cases"],
            t["cases_prevented"],
        ],
    ):
        with col:
            st.metric(label, "—", help=t["awaiting_backend"])

    st.caption(
        f"{len(selected_wards)} {t['wards']} · "
        f"{intervention_teams} · {simulation_horizon} weeks"
    )

    st.warning(t["no_backend_results"])

elif page == "dispatch_center":
    st.markdown(f"### {t['dispatch_center']}")
    st.caption(t["dispatch_desc"])

    available_wards = sorted(
        clean_data[WARD_COL].dropna().unique().tolist()
    )

    selected_dispatch_ward = st.selectbox(
        t["dispatch_ward"],
        available_wards,
        key="dispatch_ward_preview",
    )

    dispatch_format = st.radio(
        t["dispatch_format"],
        ["SMS", "WhatsApp", "PDF", "HTML"],
        horizontal=True,
        key="dispatch_format_preview",
    )

    st.markdown(f"#### {t['dispatch_preview']}")

    st.caption(
        f"{t['ward']}: {selected_dispatch_ward} · "
        f"{t['dispatch_format']}: {dispatch_format}"
    )

    st.info(t["dispatch_unavailable"])

    st.button(
        t["generate_dispatch"],
        type="primary",
        width="stretch",
        disabled=True,
    )


st.divider()

if page in ("dashboard", "ward_forecast"):
    render_risk_guide()

st.divider()

st.caption("DengueGuard AI")
st.caption(t["disclaimer"])
