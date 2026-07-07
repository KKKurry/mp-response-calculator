from __future__ import annotations

from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Keep custom transformer importable for joblib.
from model_pipeline import MODEL_FEATURES  # noqa: F401

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model" / "mp_rf_model.joblib"
METADATA_PATH = APP_DIR / "model" / "metadata.json"
CSS_PATH = APP_DIR / "assets" / "style.css"
SAMPLE_CSV_PATH = APP_DIR / "sample_input.csv"
SAMPLE_XLSX_PATH = APP_DIR / "sample_input.xlsx"

st.set_page_config(
    page_title="MP Response Prediction System",
    page_icon="MP",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource(show_spinner=False)
def load_artifact():
    artifact = joblib.load(MODEL_PATH)
    return artifact["model"], artifact["metadata"]


def load_css() -> None:
    if CSS_PATH.exists():
        st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * float(x):.{digits}f}%"


def probability_tier(prob: float) -> tuple[str, str, str, str]:
    if prob < 0.40:
        return (
            "Low likelihood",
            "MP 4–5良好反应可能性较低",
            "建议结合影像复核、病理信息和完整治疗方案进行综合判断。",
            "tier-low",
        )
    if prob < 0.70:
        return (
            "Intermediate likelihood",
            "MP 4–5良好反应可能性中等",
            "建议在MDT语境下解释该结果，并关注MRI与IHC信息的一致性。",
            "tier-mid",
        )
    return (
        "High likelihood",
        "MP 4–5良好反应可能性较高",
        "提示该患者可能存在较好的治疗反应倾向，但仍需结合临床证据链。",
        "tier-high",
    )


def normalize_batch_columns(df: pd.DataFrame) -> pd.DataFrame:
    alias = {
        "age": "年龄", "Age": "年龄", "AGE": "年龄",
        "calcification": "钙化", "Calcification": "钙化", "calc": "钙化", "Calc": "钙化",
        "curve": "曲线", "Curve": "曲线", "dce_curve": "曲线", "DCE_curve": "曲线",
        "adc": "ADC值", "ADC": "ADC值", "ADC_value": "ADC值",
        "size": "cm", "tumor_size": "cm", "Tumor_size_cm": "cm", "tumour_size_cm": "cm",
        "er": "ER", "ER_percent": "ER", "ER%": "ER",
        "pr": "PR", "PR_percent": "PR", "PR%": "PR",
        "her2": "HER2", "HER2_score": "HER2", "HER-2": "HER2",
        "ki67": "Ki-67", "Ki67": "Ki-67", "Ki67_percent": "Ki-67", "Ki-67_percent": "Ki-67",
    }
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    for src, dst in alias.items():
        if src in out.columns and dst not in out.columns:
            out[dst] = out[src]
    return out


def read_uploaded_table(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    raise ValueError("仅支持 CSV、XLSX 或 XLS 文件。")


def dataframe_to_xlsx_bytes(df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Predictions")
    return output.getvalue()


def input_to_dataframe(values: dict) -> pd.DataFrame:
    calc_text = "病灶内钙化" if values["calcification"] == "有" else "无"
    curve_text = None if values["curve"] == "未知 / Not available" else values["curve"]
    return pd.DataFrame([
        {
            "年龄": values["age"],
            "钙化": calc_text,
            "曲线": curve_text,
            "ADC值": values["adc"],
            "cm": f"{values['tumor_size']:.2f}",
            "ER": f"{values['er']}%",
            "PR": f"{values['pr']}%",
            "HER2": values["her2"],
            "Ki-67": f"{values['ki67']}%",
        }
    ])


def render_top_science_art() -> None:
    st.markdown(
        r'''
        <div class="gq-science-art" aria-label="scientific illustration banner">
          <svg viewBox="0 0 1500 190" preserveAspectRatio="none">
            <defs>
              <linearGradient id="blue" x1="0" x2="1"><stop stop-color="#1d4ed8"/><stop offset="1" stop-color="#06b6d4"/></linearGradient>
              <linearGradient id="green" x1="0" x2="1"><stop stop-color="#bef264"/><stop offset="1" stop-color="#84cc16"/></linearGradient>
              <linearGradient id="purple" x1="0" x2="1"><stop stop-color="#c4b5fd"/><stop offset="1" stop-color="#7c3aed"/></linearGradient>
              <filter id="softShadow"><feDropShadow dx="0" dy="8" stdDeviation="10" flood-opacity="0.18"/></filter>
            </defs>
            <rect width="1500" height="190" fill="#f4f0e6"/>
            <path d="M10,70 C85,5 150,142 235,68 S388,56 485,98 S640,136 745,78 S916,21 1026,88 S1202,103 1320,56 S1430,42 1490,72" fill="none" stroke="#c2bfb4" stroke-width="2" opacity=".7"/>
            <path d="M24,74 C96,25 155,128 239,78 S388,74 481,112 S638,125 746,90 S914,40 1027,101 S1205,116 1321,70 S1438,55 1485,88" fill="none" stroke="url(#blue)" stroke-width="11" stroke-linecap="round"/>
            <path d="M25,103 C96,55 155,154 239,105 S388,100 481,137 S638,151 746,117 S914,68 1027,127 S1205,144 1321,98 S1438,86 1485,118" fill="none" stroke="#0f766e" stroke-width="5" stroke-linecap="round" opacity=".85"/>
            <g class="art-labels">
              <text x="78" y="38" transform="rotate(-14 78 38)">MRI</text>
              <text x="195" y="36" transform="rotate(-18 195 36)">IHC</text>
              <text x="358" y="50" transform="rotate(-25 358 50)">ADC</text>
              <text x="630" y="42" transform="rotate(-9 630 42)">response</text>
              <text x="1186" y="44" transform="rotate(12 1186 44)">MP 4–5</text>
            </g>
            <g transform="translate(210 36)" filter="url(#softShadow)">
              <path d="M0 34 C25 -18 82 -19 106 22 C133 64 100 108 57 106 C18 105 -17 72 0 34Z" fill="url(#green)" stroke="#334155" stroke-width="1.5"/>
              <path d="M19 82 C38 30 74 84 96 26" stroke="#365314" stroke-width="6" fill="none" stroke-linecap="round"/>
            </g>
            <g transform="translate(690 77)">
              <circle cx="0" cy="0" r="22" fill="url(#purple)" stroke="#334155"/><circle cx="54" cy="0" r="22" fill="url(#purple)" stroke="#334155"/>
              <circle cx="108" cy="0" r="22" fill="url(#purple)" stroke="#334155"/><circle cx="162" cy="0" r="22" fill="url(#purple)" stroke="#334155"/>
              <path d="M-24 30 C36 56 104 56 190 30" stroke="#64748b" fill="none" stroke-width="2"/>
            </g>
            <g transform="translate(955 58)" opacity=".95">
              <g fill="#c4b5fd" stroke="#334155" stroke-width="1">
                <circle cx="0" cy="22" r="16"/><circle cx="32" cy="12" r="15"/><circle cx="64" cy="18" r="16"/><circle cx="96" cy="8" r="14"/>
                <circle cx="128" cy="16" r="15"/><circle cx="160" cy="23" r="16"/><circle cx="192" cy="13" r="15"/>
              </g>
              <path d="M-12 54 L206 54" stroke="#14b8a6" stroke-width="5" stroke-linecap="round" opacity=".75"/>
            </g>
            <g transform="translate(1280 42)">
              <path d="M50 0 C134 22 114 99 56 109 C-6 120 -28 52 50 0Z" fill="none" stroke="#2563eb" stroke-width="3"/>
              <path d="M20 34 C82 -2 130 45 86 92 C50 130 -24 91 20 34Z" fill="none" stroke="#7c3aed" stroke-width="3" opacity=".9"/>
              <path d="M24 91 C78 22 126 78 41 23" fill="none" stroke="#0f766e" stroke-width="2" opacity=".8"/>
              <text x="76" y="-8" class="art-end">clinical AI</text>
            </g>
          </svg>
        </div>
        ''',
        unsafe_allow_html=True,
    )


def render_portal_nav() -> None:
    st.markdown(
        '''
        <header class="gq-nav-shell">
          <div class="gq-brand"><div class="gq-mark">MP</div><div><strong>mpreslab</strong><span>MRI–IHC response web server</span></div></div>
          <nav class="gq-menu">
            <a href="#home">Home</a><a href="#calculator">Calculator</a><a href="#batch">Batch</a><a href="#guide">Model Guide</a><a href="#contact">Contact</a>
          </nav>
        </header>
        ''',
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        '''
        <section class="gq-hero" id="home">
          <span class="gq-new">New</span>
          <h1>MRI–IHC MP Response Calculator<br/>Miller–Payne Prediction System</h1>
          <p>Pre-treatment clinical, MRI and immunohistochemical feature analysis for estimating favorable pathological response in breast cancer.</p>
        </section>
        <div class="gq-tabs">
          <a class="active" href="#calculator">📊 Parameter Calculation</a>
          <a href="#calculator">🔎 Feature Input</a>
          <a href="#guide">📈 Model Guide</a>
          <a href="#batch">🧬 Batch Prediction</a>
        </div>
        <div class="gq-live-card">
          <div><i></i><b>Online</b><span>Running</span></div>
          <div><i class="orange"></i><b>9</b><span>Variables</span></div>
          <div><i class="purple"></i><b>MP</b><span>Endpoint</span></div>
        </div>
        ''',
        unsafe_allow_html=True,
    )


def render_input_controls() -> tuple[bool, dict]:
    st.markdown('<a id="calculator"></a>', unsafe_allow_html=True)
    st.markdown(
        '''
        <div class="section-heading">
          <div class="heading-icon">▦</div>
          <div><h2>Online Calculation</h2><p>逐项输入患者治疗前可获得变量，系统即时计算MP 4–5良好反应概率。</p></div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    with st.form("single_case_form", clear_on_submit=False):
        st.markdown('<div class="form-pill">Clinical profile</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3, gap="medium")
        with c1:
            age = st.number_input("Age / 年龄", min_value=18, max_value=95, value=52, step=1)
        with c2:
            tumor_size = st.number_input("Tumor size, cm / 肿瘤最大径", min_value=0.10, max_value=12.00, value=2.80, step=0.10, format="%.2f")
        with c3:
            adc = st.number_input("ADC value / ADC值", min_value=0.10, max_value=3.00, value=0.95, step=0.01, format="%.3f")

        st.markdown('<div class="form-pill">MRI domain</div>', unsafe_allow_html=True)
        c4, c5 = st.columns([1, 1.4], gap="large")
        with c4:
            calcification = st.radio("Calcification / 钙化", ["有", "无"], horizontal=True)
        with c5:
            curve = st.selectbox("DCE-MRI kinetic curve / 动态增强曲线", ["流出", "平台", "持续", "未知 / Not available"], index=0)

        st.markdown('<div class="form-pill">Immunohistochemistry domain</div>', unsafe_allow_html=True)
        c6, c7, c8, c9 = st.columns(4, gap="medium")
        with c6:
            er = st.number_input("ER, %", min_value=0, max_value=100, value=10, step=1)
        with c7:
            pr = st.number_input("PR, %", min_value=0, max_value=100, value=5, step=1)
        with c8:
            her2 = st.selectbox("HER2 score", [0, 1, 2, 3], index=0)
        with c9:
            ki67 = st.number_input("Ki-67, %", min_value=0, max_value=100, value=70, step=1)

        submitted = st.form_submit_button("Calculate MP 4–5 Probability  /  计算MP 4–5概率", use_container_width=True)

    values = {
        "age": age,
        "tumor_size": tumor_size,
        "adc": adc,
        "calcification": calcification,
        "curve": curve,
        "er": er,
        "pr": pr,
        "her2": her2,
        "ki67": ki67,
    }
    return submitted, values


def render_current_values(values: dict) -> None:
    rows = [
        ("Age", f"{values['age']} years"),
        ("Tumor size", f"{values['tumor_size']:.2f} cm"),
        ("ADC", f"{values['adc']:.3f}"),
        ("Calcification", values["calcification"]),
        ("Kinetic curve", values["curve"]),
        ("ER", f"{values['er']}%"),
        ("PR", f"{values['pr']}%"),
        ("HER2", str(values["her2"])),
        ("Ki-67", f"{values['ki67']}%"),
    ]
    html = '<div class="side-card"><h3>Current Input Profile</h3><p>输入变量实时概览</p><div class="mini-table">'
    for k, v in rows:
        html += f"<div><span>{k}</span><b>{v}</b></div>"
    html += "</div></div>"
    st.markdown(html, unsafe_allow_html=True)


def render_prediction(prob: float, threshold: float) -> None:
    tier_en, tier_cn, advice, tier_class = probability_tier(prob)
    predicted_group = "Favorable response / MP 4–5" if prob >= threshold else "Non-favorable response / MP 1–3"
    st.markdown(
        f'''
        <div class="result-card {tier_class}">
          <div class="result-kicker">Calculation Results</div>
          <h3>Predicted probability of MP 4–5 response</h3>
          <div class="probability">{pct(prob)}</div>
          <div class="prob-bar"><span style="width:{100*float(prob):.1f}%"></span></div>
          <div class="tier-label">{tier_en}</div>
          <p class="tier-cn">{tier_cn}</p>
          <div class="decision-box"><span>Model-predicted group</span><b>{predicted_group}</b></div>
          <p class="advice">{advice}</p>
        </div>
        ''',
        unsafe_allow_html=True,
    )


def render_empty_result() -> None:
    st.markdown(
        '''
        <div class="result-card empty-result">
          <div class="result-kicker">Calculation Results</div>
          <h3>Waiting for feature input</h3>
          <p>完成左侧变量录入并点击计算后，此处将显示MP 4–5概率、可能性分层和模型预测组别。</p>
          <div class="empty-steps"><span>1</span> Input features</div>
          <div class="empty-steps"><span>2</span> Run calculator</div>
          <div class="empty-steps"><span>3</span> Export result</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )


def render_download_single(result_df: pd.DataFrame) -> None:
    csv = result_df.to_csv(index=False).encode("utf-8-sig")
    xlsx = dataframe_to_xlsx_bytes(result_df)
    d1, d2 = st.columns(2)
    with d1:
        st.download_button("Export CSV", data=csv, file_name="single_case_prediction.csv", mime="text/csv", use_container_width=True)
    with d2:
        st.download_button("Export Excel", data=xlsx, file_name="single_case_prediction.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)


def render_batch(model, metadata) -> None:
    st.markdown('<a id="batch"></a>', unsafe_allow_html=True)
    st.markdown(
        '''
        <div class="section-heading compact"><div class="heading-icon">⇧</div><div><h2>Batch Prediction</h2><p>上传CSV或Excel，批量输出MP 4–5预测概率。</p></div></div>
        ''', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload patient table / 上传患者表格", type=["csv", "xlsx", "xls"], key="batch_upload")
    c1, c2 = st.columns(2)
    with c1:
        if SAMPLE_CSV_PATH.exists():
            st.download_button("Download CSV template", SAMPLE_CSV_PATH.read_bytes(), "sample_input.csv", "text/csv", use_container_width=True)
    with c2:
        if SAMPLE_XLSX_PATH.exists():
            st.download_button("Download Excel template", SAMPLE_XLSX_PATH.read_bytes(), "sample_input.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
    if uploaded is None:
        st.info("请上传包含年龄、钙化、曲线、ADC、肿瘤大小、ER、PR、HER2、Ki-67等字段的表格。")
        return
    try:
        raw_df = read_uploaded_table(uploaded)
        df = normalize_batch_columns(raw_df)
        probs = model.predict_proba(df)[:, 1]
        threshold = float(metadata.get("operating_threshold", 0.5))
        out = raw_df.copy()
        out["Predicted_MP4_5_probability"] = np.round(probs, 4)
        out["Prediction_group"] = np.where(probs >= threshold, "Favorable / MP 4-5", "Non-favorable / MP 1-3")
        out["Likelihood_tier"] = [probability_tier(p)[0] for p in probs]
        st.success(f"完成 {len(out)} 条样本预测。")
        st.dataframe(out, use_container_width=True, height=360)
        d1, d2 = st.columns(2)
        with d1:
            st.download_button("Export Results CSV", out.to_csv(index=False).encode("utf-8-sig"), "batch_predictions.csv", "text/csv", use_container_width=True)
        with d2:
            st.download_button("Export Results Excel", dataframe_to_xlsx_bytes(out), "batch_predictions.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
    except Exception as exc:
        st.error(f"批量预测失败：{exc}")


def render_guide() -> None:
    st.markdown('<a id="guide"></a>', unsafe_allow_html=True)
    st.markdown(
        '''
        <div class="guide-grid">
          <div class="guide-card"><h3>Supported Input Features</h3><p>Clinical profile, MRI morphology / kinetic descriptors and IHC biomarkers available before treatment.</p></div>
          <div class="guide-card"><h3>Model Output</h3><p>The output is an individualized probability estimate for Miller–Payne grade 4–5 pathological response.</p></div>
          <div class="guide-card"><h3>Research-use Notice</h3><p>This prototype is intended for research and presentation only. It is not a standalone diagnostic or treatment decision device.</p></div>
        </div>
        <div id="contact" class="footer-note">© MP Response Prediction System · Research-use clinical calculator prototype</div>
        ''',
        unsafe_allow_html=True,
    )


def main() -> None:
    load_css()
    model, metadata = load_artifact()
    threshold = float(metadata.get("operating_threshold", 0.5))

    render_top_science_art()
    render_portal_nav()
    render_hero()

    left, right = st.columns([1.48, 0.92], gap="medium")
    with left:
        submitted, values = render_input_controls()
    with right:
        render_current_values(values)
        if submitted:
            input_df = input_to_dataframe(values)
            prob = float(model.predict_proba(input_df)[0, 1])
            render_prediction(prob, threshold)
            single_out = input_df.copy()
            single_out["Predicted_MP4_5_probability"] = round(prob, 4)
            single_out["Likelihood_tier"] = probability_tier(prob)[0]
            single_out["Prediction_group"] = "Favorable / MP 4-5" if prob >= threshold else "Non-favorable / MP 1-3"
            render_download_single(single_out)
        else:
            render_empty_result()

    with st.expander("Batch prediction / 批量预测", expanded=False):
        render_batch(model, metadata)

    render_guide()


if __name__ == "__main__":
    main()
