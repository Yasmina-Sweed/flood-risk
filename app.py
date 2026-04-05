
import streamlit as st
import numpy as np
import pickle, os

st.set_page_config(page_title="Flood Risk for Farmers", page_icon="🌊",
                   layout="centered", initial_sidebar_state="collapsed")

MODEL_PATH  = "model.pkl"
SCALER_PATH = "scaler.pkl"
FEAT_PATH   = "feature_names.pkl"
LABEL_MAP   = {0: "🟢 Low Risk", 1: "🟡 Medium Risk", 2: "🔴 High Risk"}
RISK_COLORS = {0: "#EAF3DE", 1: "#FAEEDA", 2: "#FCEBEB"}
RISK_BORDER = {0: "#639922", 1: "#BA7517", 2: "#A32D2D"}

RECOMMENDATIONS = {
    0: [("✅","Keep monitoring","Your risk is low — stay alert during heavy rain seasons."),
        ("🌿","Maintain vegetation","Keep trees and plants around your land; they absorb excess water."),
        ("📱","Sign up for alerts","Register for local weather and flood-warning services.")],
    1: [("🏗️","Improve drainage","Clear drainage channels and create small ditches around crop rows."),
        ("🌳","Plant buffer trees","A tree belt on the flood-prone side slows water significantly."),
        ("🏠","Raise storage areas","Store seeds, tools, and food above the potential flood level."),
        ("📋","Make an emergency plan","Identify safe routes and shelters with your family now.")],
    2: [("🚨","Act now — high risk","Your farm is at significant risk. Start protective actions today."),
        ("🌊","Build earthen barriers","Small embankments (30-50 cm) can block flash flood water."),
        ("🌱","Flood-tolerant crops","Consider rice, taro, or other varieties suited to wet conditions."),
        ("📦","Emergency supply kit","Prepare waterproof bags, first aid, 3-day food, key documents."),
        ("🤝","Contact authorities","Talk to your local agricultural office about flood subsidies.")],
}

def inject_css():
    st.markdown("""
    <style>
        .block-container { max-width: 420px; padding: 1.5rem 1rem 3rem; }
        div.stButton > button { width:100%; background:white; border:1px solid #e0e0e0;
            border-radius:12px; padding:14px 16px; font-size:15px; text-align:left;
            margin-bottom:8px; }
        div.stButton > button:hover { border-color:#4A90D9; background:#EBF4FF; color:#1a5fa8; }
        .fact-box { background:#FFF8F0; border:1px solid #F5D5B0; border-radius:12px;
            padding:14px 16px; margin:16px 0; font-size:13px; color:#5a4030; }
        .fact-box ul { margin:6px 0 0 16px; } .fact-box li { margin-bottom:4px; }
        .risk-card { border-radius:16px; padding:20px; text-align:center; margin:16px 0; }
        .risk-score-big { font-size:48px; font-weight:600; line-height:1; }
        .risk-badge { display:inline-block; font-size:14px; font-weight:600;
            padding:4px 14px; border-radius:20px; margin-top:6px; }
        .rec-card { background:#f7f7f7; border-radius:10px; padding:12px 14px;
            margin-bottom:8px; display:flex; gap:10px; align-items:flex-start; }
        .rec-icon { font-size:20px; flex-shrink:0; }
        .rec-title { font-size:13px; font-weight:600; color:#1a1a1a; }
        .rec-text { font-size:12px; color:#555; line-height:1.45; }
        .app-header { display:flex; justify-content:space-between; align-items:center;
            padding-bottom:12px; border-bottom:1px solid #eee; margin-bottom:20px; }
        .app-title { font-size:15px; font-weight:600; }
        .lang-badge { font-size:11px; font-weight:600; border:1px solid #ccc;
            border-radius:6px; padding:2px 8px; color:#666; }
        #MainMenu, footer, header { visibility:hidden; }
    </style>""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load .pkl files once per session."""
    if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, FEAT_PATH]):
        return None, None, None
    with open(MODEL_PATH,  "rb") as f: model    = pickle.load(f)
    with open(SCALER_PATH, "rb") as f: scaler   = pickle.load(f)
    with open(FEAT_PATH,   "rb") as f: features = pickle.load(f)
    return model, scaler, features

def predict_risk(model, scaler, feature_names, user_inputs):
    """Build feature vector, scale, predict. Return (label int, confidence %)."""
    row = {f: user_inputs.get(f, 5) for f in feature_names}
    X   = np.array([[row[f] for f in feature_names]])
    X_s = scaler.transform(X)
    label     = int(model.predict(X_s)[0])
    proba_pct = int(model.predict_proba(X_s)[0][label] * 100)
    return label, proba_pct

def rule_based_fallback(user_inputs):
    """Weighted fallback when model files are missing."""
    score  = (user_inputs.get("MonsoonIntensity", 5) * 3
            + user_inputs.get("Deforestation",    5) * 2
            + user_inputs.get("Urbanization",     5) * 2
            + user_inputs.get("_region_bonus",    3))
    pct = int(score / (9*3 + 9*2 + 9*2 + 4) * 100)
    if pct < 40: return 0, pct
    if pct < 68: return 1, pct
    return 2, pct

def render_header():
    st.markdown(''''<div class="app-header">
        <span class="app-title">🌊 Flood Risk</span>
        <span class="lang-badge">EN</span></div>''''', unsafe_allow_html=True)

def render_progress(step, total=5):
    st.progress(step / total)

def render_fact_box(items, title="Did you know?"):
    li = "".join(f"<li>{i}</li>" for i in items)
    st.markdown(f'<div class="fact-box"><strong>{title}</strong><ul>{li}</ul></div>',
                unsafe_allow_html=True)

def render_option_buttons(options, key):
    selected = st.session_state.get(key)
    for label, value in options:
        prefix = "✓  " if selected == value else "     "
        if st.button(prefix + label, key=f"btn_{key}_{value}"):
            st.session_state[key] = value
            st.rerun()
    return st.session_state.get(key)

def render_recommendations(risk_level):
    for icon, title, text in RECOMMENDATIONS[risk_level]:
        st.markdown(f'<div class="rec-card"><div class="rec-icon">{icon}</div>'
                    f'<div><div class="rec-title">{title}</div>'
                    f'<div class="rec-text">{text}</div></div></div>',
                    unsafe_allow_html=True)

def render_risk_result(risk_level, score_pct):
    color  = RISK_COLORS[risk_level]
    border = RISK_BORDER[risk_level]
    label  = LABEL_MAP[risk_level]
    st.markdown(
        f'<div class="risk-card" style="background:{color}; border:3px solid {border};">'
        f'<div class="risk-score-big" style="color:{border};">{score_pct}%</div>'
        f'<div class="risk-badge" style="background:{border}; color:white;">{label}</div>'
        f'</div>', unsafe_allow_html=True)

# ── Screens ────────────────────────────────────────────────────────────────────

def screen_who_are_you():
    render_header()
    st.subheader("Is your area at risk?")
    st.caption("Answer some questions")
    st.markdown("**Who are you?**")
    selected = render_option_buttons([
        ("🏠  City resident — I live in a town or city", "resident"),
        ("🌱  Farmer — I work the land",                  "farmer"),
        ("👔  City mayor — I represent a community",      "mayor"),
    ], key="role")
    render_fact_box(["Floods affect 1 in 5 people worldwide every year",
                     "Farmers are the most vulnerable to flood damage",
                     "Early warning can reduce flood damage by up to 30%"])
    if selected:
        if st.button("Continue →", key="go_s2", type="primary"):
            st.session_state.step = 2; st.rerun()

def screen_region():
    render_header(); render_progress(1)
    st.subheader("Where is your farm?")
    st.caption("Choose your region for accurate advice")
    selected = render_option_buttons([
        ("🌍  Africa — Sahel, East Africa, West Africa", "africa"),
        ("🌏  Asia — South Asia, Southeast Asia",        "asia"),
        ("🌎  South America — Amazon region, Andes",     "southamerica"),
    ], key="region")
    render_fact_box(["Africa: Sudden flash floods after dry seasons are most dangerous",
                     "Asia: Monsoon season brings the highest flood risk each year",
                     "South America: Amazon flooding peaks between January and April"],
                    title="Your region matters")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("← Back", key="back_s2"): st.session_state.step=1; st.rerun()
    with c2:
        if selected and st.button("Continue →", key="go_s3", type="primary"):
            st.session_state.step=3; st.rerun()

def screen_rainfall():
    render_header(); render_progress(2)
    st.subheader("How often does rain fall on your land?")
    st.caption("Protect your farm and family")
    selected = render_option_buttons([
        ("Very little / None", 1), ("Some rain", 5), ("Heavy rains", 9)
    ], key="rain")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("← Back", key="back_s3"): st.session_state.step=2; st.rerun()
    with c2:
        if selected is not None and st.button("Continue →", key="go_s4", type="primary"):
            st.session_state.step=4; st.rerun()

def screen_trees():
    render_header(); render_progress(3)
    st.subheader("Are there trees and vegetation around your farm?")
    st.caption("Protect your farm and family")
    selected = render_option_buttons([
        ("Very few / None", 9), ("Some trees", 5), ("Many trees", 1)
    ], key="trees")
    render_fact_box(["Africa: Sudden flash floods after dry seasons are most dangerous",
                     "Asia: Monsoon season brings the highest flood risk each year",
                     "South America: Amazon flooding peaks between January and April"],
                    title="Your region matters")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("← Back", key="back_s4"): st.session_state.step=3; st.rerun()
    with c2:
        if selected is not None and st.button("Continue →", key="go_s5", type="primary"):
            st.session_state.step=5; st.rerun()

def screen_drainage():
    render_header(); render_progress(4)
    st.subheader("How is the ground near your land?")
    st.caption("Protect your farm and family")
    selected = render_option_buttons([
        ("Mostly natural soil / fields", 1),
        ("Mix of fields and roads", 5),
        ("Mostly concrete / urban area", 9),
    ], key="drainage")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("← Back", key="back_s5"): st.session_state.step=4; st.rerun()
    with c2:
        if selected is not None and st.button("See my flood risk →", key="go_result", type="primary"):
            st.session_state.step=6; st.rerun()

def screen_results():
    render_header(); render_progress(5)
    region_bonus = {"africa": 4, "asia": 3, "southamerica": 2}
    user_inputs = {
        "MonsoonIntensity":   st.session_state.get("rain",     5),
        "Deforestation":      st.session_state.get("trees",    5),
        "Urbanization":       st.session_state.get("drainage", 5),
        "TopographyDrainage": st.session_state.get("drainage", 5),
        "DrainageSystems":    10 - st.session_state.get("drainage", 5),
        "ClimateChange":      5,
        "WetlandLoss":        st.session_state.get("trees",    5),
        "_region_bonus":      region_bonus.get(st.session_state.get("region", "asia"), 3),
    }
    model, scaler, feature_names = load_model()
    if model is not None:
        risk_level, score_pct = predict_risk(model, scaler, feature_names, user_inputs)
        source = "Random Forest ML model"
    else:
        risk_level, score_pct = rule_based_fallback(user_inputs)
        source = "rule-based estimate (train model for ML predictions)"
    render_risk_result(risk_level, score_pct)
    headlines = {0:"Your farm is relatively safe", 1:"Take precautions this season", 2:"Your farm needs protection now"}
    descs = {0:"Flood risk is manageable. Keep monitoring your land.",
             1:"Moderate risk. Simple measures can significantly reduce damage.",
             2:"High risk. Immediate action can protect your crops and family."}
    st.markdown(f"### {headlines[risk_level]}")
    st.caption(descs[risk_level])
    st.caption(f"_Source: {source}_")
    st.markdown("---")
    st.markdown("**Recommended actions**")
    render_recommendations(risk_level)
    st.markdown("---")
    if st.button("🔄  Start over", key="restart"):
        for k in ["step","role","region","rain","trees","drainage"]:
            st.session_state.pop(k, None)
        st.rerun()

SCREENS = {1: screen_who_are_you, 2: screen_region, 3: screen_rainfall,
           4: screen_trees, 5: screen_drainage, 6: screen_results}

def main():
    inject_css()
    if "step" not in st.session_state:
        st.session_state.step = 1
    SCREENS.get(st.session_state.step, screen_who_are_you)()

if __name__ == "__main__":
    main()
