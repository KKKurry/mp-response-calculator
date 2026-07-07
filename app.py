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
    page_title="MRI–IHC MP Response Calculator",
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
            "该结果提示良好病理反应倾向较弱，建议结合影像复核、病理信息、治疗方案和MDT意见综合判断。",
            "tier-low",
        )
    if prob < 0.70:
        return (
            "Intermediate likelihood",
            "MP 4–5良好反应可能性中等",
            "该结果处于中间区间，建议重点关注MRI动态增强曲线、ADC与IHC指标之间是否存在一致性。",
            "tier-mid",
        )
    return (
        "High likelihood",
        "MP 4–5良好反应可能性较高",
        "该结果提示患者可能具有较好的治疗反应倾向，但仍需结合临床证据链进行最终判断。",
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
          <svg viewBox="0 0 1600 180" preserveAspectRatio="none">
            <defs>
              <linearGradient id="blue" x1="0" x2="1"><stop stop-color="#2563eb"/><stop offset="1" stop-color="#06b6d4"/></linearGradient>
              <linearGradient id="green" x1="0" x2="1"><stop stop-color="#bef264"/><stop offset="1" stop-color="#84cc16"/></linearGradient>
              <linearGradient id="purple" x1="0" x2="1"><stop stop-color="#ddd6fe"/><stop offset="1" stop-color="#7c3aed"/></linearGradient>
            </defs>
            <rect width="1600" height="180" fill="#f4f0e6"/>
            <path d="M20,68 C110,5 185,142 288,70 S468,49 578,101 S760,137 892,76 S1088,24 1228,89 S1430,102 1570,61" fill="none" stroke="#c9c3b6" stroke-width="2" opacity=".75"/>
            <path d="M26,78 C116,31 188,130 292,82 S468,69 574,115 S760,126 892,92 S1086,48 1228,104 S1434,118 1568,88" fill="none" stroke="url(#blue)" stroke-width="10" stroke-linecap="round"/>
            <path d="M26,104 C116,57 188,154 292,108 S468,99 574,139 S760,151 892,119 S1086,78 1228,131 S1434,147 1568,118" fill="none" stroke="#0f766e" stroke-width="4" stroke-linecap="round" opacity=".85"/>
            <g class="art-labels">
              <text x="76" y="40" transform="rotate(-13 76 40)">MRI</text>
              <text x="226" y="38" transform="rotate(-17 226 38)">IHC</text>
              <text x="396" y="52" transform="rotate(-24 396 52)">ADC</text>
              <text x="705" y="43" transform="rotate(-8 705 43)">response</text>
              <text x="1250" y="46" transform="rotate(10 1250 46)">MP 4–5</text>
            </g>
            <g transform="translate(232 37)">
              <path d="M0 34 C26 -18 82 -18 108 22 C136 64 101 108 58 106 C18 105 -18 72 0 34Z" fill="url(#green)" stroke="#334155" stroke-width="1.5"/>
              <path d="M19 82 C38 30 75 84 98 26" stroke="#365314" stroke-width="6" fill="none" stroke-linecap="round"/>
            </g>
            <g transform="translate(760 76)">
              <circle cx="0" cy="0" r="21" fill="url(#purple)" stroke="#334155"/><circle cx="56" cy="0" r="21" fill="url(#purple)" stroke="#334155"/>
              <circle cx="112" cy="0" r="21" fill="url(#purple)" stroke="#334155"/><circle cx="168" cy="0" r="21" fill="url(#purple)" stroke="#334155"/>
              <path d="M-24 30 C44 57 114 57 198 30" stroke="#64748b" fill="none" stroke-width="2"/>
            </g>
            <g transform="translate(1030 61)" opacity=".95">
              <g fill="#c4b5fd" stroke="#334155" stroke-width="1">
                <circle cx="0" cy="22" r="16"/><circle cx="35" cy="12" r="15"/><circle cx="70" cy="18" r="16"/><circle cx="105" cy="8" r="14"/>
                <circle cx="140" cy="16" r="15"/><circle cx="175" cy="23" r="16"/><circle cx="210" cy="13" r="15"/>
              </g>
              <path d="M-14 54 L226 54" stroke="#14b8a6" stroke-width="5" stroke-linecap="round" opacity=".75"/>
            </g>
            <g transform="translate(1370 42)">
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
        <section class="gq-hero refined" id="home">
          <div class="hero-main">
            <span class="gq-new">Research-use clinical AI calculator</span>
            <h1>MRI–IHC MP Response Calculator</h1>
            <h2>Miller–Payne Prediction System for Breast Cancer</h2>
            <p>A pre-treatment web calculator for estimating favorable pathological response using clinical variables, MRI phenotypes and immunohistochemical biomarkers.</p>
            <div class="hero-badges">
              <span><i>🎯</i><b>Endpoint</b> MP 4–5 response</span>
              <span><i>🧬</i><b>Input</b> Clinical + MRI + IHC</span>
              <span><i>📈</i><b>Output</b> Probability estimate</span>
              <span><i>🔬</i><b>Use</b> Research-only</span>
            </div>
          </div>
          <div class="hero-workflow">
            <div class="workflow-title">Prediction workflow</div>
            <div class="workflow-step"><span>1</span><b>Enter pre-treatment features</b><em>年龄、MRI、IHC</em></div>
            <div class="workflow-step"><span>2</span><b>Calculate individualized probability</b><em>MP 4–5概率估计</em></div>
            <div class="workflow-step"><span>3</span><b>Review interpretation and export</b><em>分层解释与结果导出</em></div>
          </div>
        </section>
        <div class="gq-tabs refined-tabs">
          <a class="active" href="#calculator">📊 Parameter Calculation</a>
          <a href="#feature-notes">🔎 Feature Notes</a>
          <a href="#guide">📈 Interpretation Guide</a>
          <a href="#batch">🧬 Batch Prediction</a>
        </div>
        ''',
        unsafe_allow_html=True,
    )


def render_research_summary() -> None:
    st.markdown(
        '''
        <div class="summary-grid enriched-summary">
          <div class="summary-card"><small>Clinical question</small><b>Pre-treatment prediction of MP 4–5 response</b><span>治疗前预测良好病理反应，辅助研究分层与模型展示。</span></div>
          <div class="summary-card"><small>Feature scope</small><b>9 clinically available features</b><span>年龄、肿瘤大小、ADC、钙化、曲线、ER、PR、HER2、Ki-67。</span></div>
          <div class="summary-card"><small>Calculation logic</small><b>Feature input → probability → tier</b><span>逐项录入变量后输出个体化概率、低/中/高分层与预测组别。</span></div>
          <div class="summary-card"><small>Clinical boundary</small><b>Research-use decision support prototype</b><span>不替代医生判断、MDT讨论或标准病理评估。</span></div>
        </div>
        ''', unsafe_allow_html=True
    )


def render_feature_notes() -> None:
    st.markdown(
        '''
        <div class="feature-notes" id="feature-notes">
          <div class="note-title"><span>Feature Dictionary</span><b>变量录入说明</b><p>所有字段均限定为治疗前可获得信息，避免治疗后病理或结局相关变量进入模型。</p></div>
          <div class="note-grid four">
            <div><b>Clinical profile</b><p>年龄与肿瘤最大径反映患者基础状态和病灶负荷。</p></div>
            <div><b>MRI domain</b><p>ADC、钙化和DCE-MRI动态增强曲线用于描述治疗前影像表型。</p></div>
            <div><b>IHC biomarkers</b><p>ER、PR、HER2和Ki-67反映受体状态和肿瘤增殖活性。</p></div>
            <div><b>Output interpretation</b><p>结果为MP 4–5概率估计，并通过低/中/高分层辅助阅读。</p></div>
          </div>
        </div>
        ''', unsafe_allow_html=True
    )


def render_input_controls() -> tuple[bool, dict]:
    st.markdown('<a id="calculator"></a>', unsafe_allow_html=True)
    st.markdown(
        '''
        <div class="section-heading spacious">
          <div class="heading-icon">▦</div>
          <div><span class="section-eyebrow">Individual calculator</span><h2>Online Calculation</h2><p>逐项输入治疗前可获得变量。右侧面板会同步展示当前输入，并在计算后给出MP 4–5概率、分层解释和模型预测组别。</p></div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    with st.form("single_case_form", clear_on_submit=False):
        st.markdown('<div class="form-section-title"><span>01</span><div><b>Clinical profile</b><em>基础临床与病灶负荷 · Basic clinical and lesion-level characteristics</em></div></div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3, gap="large")
        with c1:
            age = st.number_input("Age / 年龄", min_value=18, max_value=95, value=52, step=1, help="患者治疗前年龄。")
        with c2:
            tumor_size = st.number_input("Tumor size, cm / 肿瘤最大径", min_value=0.10, max_value=12.00, value=2.80, step=0.10, format="%.2f", help="治疗前MRI或临床记录中的病灶最大径。")
        with c3:
            adc = st.number_input("ADC value / ADC值", min_value=0.10, max_value=3.00, value=0.95, step=0.01, format="%.3f", help="治疗前MRI ADC值。")

        st.markdown('<div class="form-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="form-section-title"><span>02</span><div><b>MRI domain</b><em>治疗前影像表型 · Pre-treatment imaging phenotype</em></div></div>', unsafe_allow_html=True)
        c4, c5 = st.columns([0.95, 1.35], gap="large")
        with c4:
            calcification = st.radio("Calcification / 钙化", ["有", "无"], horizontal=True, help="病灶内是否存在钙化描述。")
        with c5:
            curve = st.selectbox("DCE-MRI kinetic curve / 动态增强曲线", ["流出", "平台", "持续", "未知 / Not available"], index=0, help="动态增强曲线类型。")

        st.markdown('<div class="form-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="form-section-title"><span>03</span><div><b>Immunohistochemistry domain</b><em>治疗前免疫组化指标 · Biomarker expression profile</em></div></div>', unsafe_allow_html=True)
        c6, c7, c8, c9 = st.columns(4, gap="medium")
        with c6:
            er = st.number_input("ER, %", min_value=0, max_value=100, value=10, step=1)
        with c7:
            pr = st.number_input("PR, %", min_value=0, max_value=100, value=5, step=1)
        with c8:
            her2 = st.selectbox("HER2 score", [0, 1, 2, 3], index=0)
        with c9:
            ki67 = st.number_input("Ki-67, %", min_value=0, max_value=100, value=70, step=1)

        st.markdown('<div class="form-footnote"><b>Input boundary</b><br/>请仅输入治疗前即可获得的信息；治疗后MRI、术后病理或任何直接反映MP结局的变量不应输入。</div>', unsafe_allow_html=True)
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


def _current_profile_html(values: dict) -> str:
    quick = [
        ("Age", f"{values['age']}y"),
        ("Tumor", f"{values['tumor_size']:.2f} cm"),
        ("ADC", f"{values['adc']:.3f}"),
    ]
    rows = [
        ("Calcification", values["calcification"]),
        ("Kinetic curve", values["curve"]),
        ("ER", f"{values['er']}%"),
        ("PR", f"{values['pr']}%"),
        ("HER2", str(values["her2"])),
        ("Ki-67", f"{values['ki67']}%"),
    ]
    html = '<div class="profile-cards">'
    for k, v in quick:
        html += f'<div><small>{k}</small><b>{v}</b></div>'
    html += '</div><div class="mini-table compact">'
    for k, v in rows:
        html += f"<div><span>{k}</span><b>{v}</b></div>"
    html += '</div>'
    return html


def _result_html(prob: float | None, threshold: float) -> str:
    if prob is None:
        return '''
        <div class="result-card empty-result refined-empty">
          <div class="result-kicker">Calculation Results</div>
          <h3>Awaiting calculation</h3>
          <p>输入治疗前特征后，点击计算即可生成个体化MP 4–5良好病理反应概率。</p>
          <div class="skeleton-donut"></div>
          <div class="skeleton-line wide"></div>
          <div class="skeleton-line mid"></div>
          <div class="empty-steps"><span>1</span> Input features</div>
          <div class="empty-steps"><span>2</span> Calculate probability</div>
          <div class="empty-steps"><span>3</span> Export result</div>
        </div>
        '''
    tier_en, tier_cn, advice, tier_class = probability_tier(prob)
    predicted_group = "Favorable response / MP 4–5" if prob >= threshold else "Non-favorable response / MP 1–3"
    return f'''
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
        '''


def render_right_dashboard(values: dict, prob: float | None, threshold: float) -> None:
    st.markdown(
        f'''
        <aside class="right-control-panel">
          <div class="side-card input-profile">
            <div class="card-eyebrow">Current input</div>
            <h3>Current Input Profile</h3>
            <p>输入变量实时概览，用于核对本次单病例预测参数。</p>
            {_current_profile_html(values)}
          </div>
          {_result_html(prob, threshold)}
          <div class="interpret-panel">
            <h3>Clinical interpretation</h3>
            <p>模型输出的是MP 4–5良好反应的概率估计，而不是确定性诊断。建议结合MRI复核、IHC一致性、治疗方案和MDT意见解释。</p>
            <div class="risk-chips"><span>Low</span><span>Intermediate</span><span>High</span></div>
          </div>
        </aside>
        ''',
        unsafe_allow_html=True,
    )


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
    html = '<div class="side-card input-profile"><div class="card-eyebrow">Current input</div><h3>Current Input Profile</h3><p>输入变量实时概览，用于核对本次单病例预测参数。</p><div class="mini-table">'
    for k, v in rows:
        html += f"<div><span>{k}</span><b>{v}</b></div>"
    html += "</div></div>"
    st.markdown(html, unsafe_allow_html=True)


def render_interpretation_panel() -> None:
    st.markdown(
        '''
        <div class="interpret-panel">
          <h3>Clinical interpretation</h3>
          <p>模型输出的是MP 4–5良好反应的概率估计，而不是确定性诊断。建议结合MRI复核、IHC一致性、治疗方案和MDT意见解释。</p>
          <ul>
            <li><b>Low:</b> 良好反应可能性偏低</li>
            <li><b>Intermediate:</b> 需结合多模态证据综合判断</li>
            <li><b>High:</b> 良好病理反应倾向较高</li>
          </ul>
        </div>
        ''', unsafe_allow_html=True
    )


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
          <h3>Awaiting calculation</h3>
          <p>完成左侧变量录入并点击计算后，此处将显示MP 4–5概率、可能性分层和模型预测组别。</p>
          <div class="empty-steps"><span>1</span> Input pre-treatment features</div>
          <div class="empty-steps"><span>2</span> Calculate individualized probability</div>
          <div class="empty-steps"><span>3</span> Export single-case result</div>
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
        <div class="section-heading compact"><div class="heading-icon">⇧</div><div><h2>Batch Prediction</h2><p>上传CSV或Excel，批量输出MP 4–5预测概率，并导出结果表格。</p></div></div>
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
        <div class="guide-title"><span>Model documentation</span><h2>How to use and interpret this calculator</h2></div>
        <div class="guide-grid enriched four-guide">
          <div class="guide-card"><h3>1. Supported input features</h3><p>输入变量限定为治疗前可获得的临床资料、MRI特征和免疫组化指标，避免纳入治疗后信息。</p></div>
          <div class="guide-card"><h3>2. Model output</h3><p>输出为MP 4–5良好病理反应的个体化概率估计，并给出低、中、高可能性分层。</p></div>
          <div class="guide-card"><h3>3. Batch data format</h3><p>支持CSV与Excel格式批量预测，结果表中自动追加概率、预测组别与分层标签。</p></div>
          <div class="guide-card"><h3>4. Clinical boundary</h3><p>该网页为科研和展示用途，不构成独立诊断工具，也不替代临床医生或MDT决策。</p></div>
        </div>
        <div id="contact" class="footer-note">© MP Response Prediction System · MRI–IHC research-use clinical calculator prototype</div>
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
    render_research_summary()

    left, right = st.columns([1.45, 0.95], gap="large")
    with left:
        submitted, values = render_input_controls()
        render_feature_notes()
    prob = None
    single_out = None
    if submitted:
        input_df = input_to_dataframe(values)
        prob = float(model.predict_proba(input_df)[0, 1])
        single_out = input_df.copy()
        single_out["Predicted_MP4_5_probability"] = round(prob, 4)
        single_out["Likelihood_tier"] = probability_tier(prob)[0]
        single_out["Prediction_group"] = "Favorable / MP 4-5" if prob >= threshold else "Non-favorable / MP 1-3"
    with right:
        render_right_dashboard(values, prob, threshold)
        if single_out is not None:
            render_download_single(single_out)

    with st.expander("Batch prediction / 批量预测", expanded=False):
        render_batch(model, metadata)

    render_guide()


if __name__ == "__main__":
    main()
