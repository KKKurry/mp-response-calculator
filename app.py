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
CSS_PATH = APP_DIR / "assets" / "style.css"
SAMPLE_CSV_PATH = APP_DIR / "sample_input.csv"
SAMPLE_XLSX_PATH = APP_DIR / "sample_input.xlsx"

st.set_page_config(
    page_title="MP Response Feature Calculator",
    page_icon="◇",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource
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
            "建议结合影像复核、病理信息和治疗方案进行综合判断。",
            "tier-low",
        )
    if prob < 0.70:
        return (
            "Intermediate likelihood",
            "MP 4–5良好反应可能性中等",
            "建议在MDT语境下解释，并关注MRI与IHC信息是否一致。",
            "tier-mid",
        )
    return (
        "High likelihood",
        "MP 4–5良好反应可能性较高",
        "提示可能存在较好的治疗反应倾向，但仍需结合完整临床证据链。",
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
    curve_text = None if values["curve"] == "未知" else values["curve"]
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


def render_top_nav() -> None:
    st.markdown(
        """
        <div class="topbar">
          <div class="topbar-brand">
            <span class="brand-mark">MP</span>
            <div>
              <b>MRI–IHC MP Response Calculator</b>
              <small>Feature-based prediction interface</small>
            </div>
          </div>
          <div class="topbar-links">
            <a href="#calculator">Calculator</a>
            <a href="#batch">Batch</a>
            <a href="#guide">Guide</a>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <section class="hero feature-hero">
          <div class="hero-main">
            <div class="journal-label">Research-use clinical prediction calculator</div>
            <h1>输入治疗前特征，<br><span>预测MP 4–5良好反应概率</span></h1>
            <p class="hero-lead">
              本网页聚焦一个核心功能：逐项输入患者临床、MRI和免疫组化特征，
              模型即时输出 Miller–Payne 4–5级良好病理反应的个体化概率与分层解释。
            </p>
            <div class="hero-cta-row">
              <a class="btn-primary" href="#calculator">进入特征预测</a>
              <a class="btn-ghost" href="#guide">查看变量说明</a>
            </div>
          </div>
          <div class="hero-panel calculator-map">
            <div class="panel-caption">Prediction workflow</div>
            <div class="workflow-line"><b>1</b><span>输入年龄、MRI和IHC特征</span></div>
            <div class="workflow-line"><b>2</b><span>点击 Calculate probability</span></div>
            <div class="workflow-line"><b>3</b><span>获得MP 4–5概率与分层</span></div>
            <div class="workflow-line"><b>4</b><span>导出或记录单病例结果</span></div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_input_form() -> tuple[bool, dict]:
    st.markdown("<a id='calculator'></a>", unsafe_allow_html=True)
    st.markdown("<div class='section-heading'><span>01</span><h2>Feature input calculator</h2></div>", unsafe_allow_html=True)
    st.markdown(
        "<p class='section-lead'>逐项录入治疗前可获得变量。点击预测后，右侧将显示MP 4–5良好反应概率、可能性分层和模型预测组别。</p>",
        unsafe_allow_html=True,
    )

    with st.form("feature_prediction_form", clear_on_submit=False):
        st.markdown("<div class='form-panel-title'>Clinical profile</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("年龄 / Age", min_value=18, max_value=95, value=52, step=1)
        with c2:
            tumor_size = st.number_input("肿瘤最大径 / Tumor size, cm", min_value=0.10, max_value=12.00, value=2.80, step=0.10, format="%.2f")
        with c3:
            adc = st.number_input("ADC值 / ADC value", min_value=0.10, max_value=3.00, value=0.95, step=0.01, format="%.3f")

        st.markdown("<div class='form-panel-title'>MRI domain</div>", unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        with m1:
            calcification = st.radio("钙化 / Calcification", ["有", "无"], horizontal=True)
        with m2:
            curve = st.selectbox("DCE-MRI动态增强曲线 / Kinetic curve", ["流出", "平台", "持续", "未知"], index=0)

        st.markdown("<div class='form-panel-title'>Immunohistochemistry domain</div>", unsafe_allow_html=True)
        i1, i2, i3, i4 = st.columns(4)
        with i1:
            er = st.slider("ER, %", min_value=0, max_value=100, value=10, step=1)
        with i2:
            pr = st.slider("PR, %", min_value=0, max_value=100, value=5, step=1)
        with i3:
            her2 = st.select_slider("HER2 score", options=["0", "1+", "2+", "3+"], value="3+")
        with i4:
            ki67 = st.slider("Ki-67, %", min_value=0, max_value=100, value=70, step=1)

        st.markdown("<div class='submit-spacer'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Calculate probability / 计算MP 4–5概率", type="primary", use_container_width=True)

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


def render_result_card(prob: float, threshold: float, single_df: pd.DataFrame) -> None:
    predicted = "MP 4–5" if prob >= threshold else "MP 1–3"
    tier_en, tier_cn, recommendation, tier_class = probability_tier(prob)
    width = int(max(1, min(100, round(prob * 100))))
    st.markdown(
        f"""
        <div class="result-shell {tier_class}">
          <div class="result-topline">Predicted probability of favourable pathological response</div>
          <div class="result-probability">{pct(prob, 1)}</div>
          <div class="result-gauge"><div style="width:{width}%"></div></div>
          <div class="result-interpretation">
            <h3>{tier_en}</h3>
            <p>{tier_cn}</p>
            <p class="recommendation">{recommendation}</p>
          </div>
          <div class="result-divider"></div>
          <div class="classification-row">
            <span>Model-assigned MP group</span><b>{predicted}</b>
          </div>
          <div class="result-note">Research use only · Not a standalone diagnostic device</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("查看本次输入特征 / Input features", expanded=False):
        st.dataframe(single_df, use_container_width=True, hide_index=True)

    result_df = single_df.copy()
    result_df.insert(0, "MP_4_5_probability", round(float(prob), 4))
    result_df.insert(1, "Probability_percent", pct(prob, 1))
    result_df.insert(2, "Likelihood_tier", tier_en)
    result_df.insert(3, "Predicted_MP_group", predicted)
    st.download_button(
        "下载本病例预测结果 / Download result",
        result_df.to_csv(index=False, encoding="utf-8-sig"),
        file_name="single_case_mp_prediction.csv",
        mime="text/csv",
        use_container_width=True,
    )


def render_pending_card() -> None:
    st.markdown(
        """
        <div class="pending-card calculator-pending">
          <div class="pending-mark">◇</div>
          <h3>等待输入特征并运行预测</h3>
          <p>左侧表单填写完成后，点击计算按钮；这里将显示个体化MP 4–5概率、分层解释和预测组别。</p>
          <div class="pending-list">
            <div><span></span>Probability of MP 4–5</div>
            <div><span></span>Low / Intermediate / High likelihood</div>
            <div><span></span>Model-assigned MP group</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_feature_summary(values: dict) -> None:
    rows = [
        ["Age", f"{values['age']} years"],
        ["Tumor size", f"{values['tumor_size']:.2f} cm"],
        ["ADC", f"{values['adc']:.3f}"],
        ["Calcification", values["calcification"]],
        ["Kinetic curve", values["curve"]],
        ["ER", f"{values['er']}%"],
        ["PR", f"{values['pr']}%"],
        ["HER2", values["her2"]],
        ["Ki-67", f"{values['ki67']}%"],
    ]
    st.markdown("""
        <div class="current-feature-head">
          <div class="eyebrow">Current values</div>
          <h3>当前输入概览</h3>
        </div>
        """, unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(rows, columns=["Feature", "Value"]), hide_index=True, use_container_width=True)


def recognized_columns_message(df: pd.DataFrame) -> str:
    normalized = normalize_batch_columns(df)
    expected = ["年龄", "钙化", "曲线", "ADC值", "cm", "ER", "PR", "HER2", "Ki-67"]
    present = [c for c in expected if c in normalized.columns]
    missing = [c for c in expected if c not in normalized.columns]
    msg = f"已识别字段：{', '.join(present) if present else '暂无'}"
    if missing:
        msg += f"；未识别字段：{', '.join(missing)}。未识别字段会按模型缺失值策略处理。"
    return msg


def render_batch_prediction(model, threshold: float) -> None:
    st.markdown("<a id='batch'></a>", unsafe_allow_html=True)
    st.markdown("<div class='section-heading muted-section'><span>02</span><h2>Batch prediction / 批量预测</h2></div>", unsafe_allow_html=True)
    st.markdown("<p class='section-lead'>单病例特征输入是主功能；批量预测作为研究整理和论文补充分析的辅助功能保留。</p>", unsafe_allow_html=True)

    with st.expander("展开批量上传模块", expanded=False):
        col1, col2 = st.columns([0.68, 0.32], gap="large")
        with col1:
            uploaded = st.file_uploader("上传 CSV / XLSX / XLS", type=["csv", "xlsx", "xls"], key="batch_file")
            if uploaded is not None:
                try:
                    batch = read_uploaded_table(uploaded)
                    st.info(recognized_columns_message(batch))
                    batch_norm = normalize_batch_columns(batch)
                    probs = model.predict_proba(batch_norm)[:, 1]
                    tiers = [probability_tier(p)[0] for p in probs]
                    result = batch.copy()
                    result.insert(0, "MP_4_5_probability", [round(float(p), 4) for p in probs])
                    result.insert(1, "Probability_percent", [pct(p, 1) for p in probs])
                    result.insert(2, "Likelihood_tier", tiers)
                    result.insert(3, "Predicted_MP_group", np.where(probs >= threshold, "MP 4-5", "MP 1-3"))
                    st.success("预测完成。")
                    st.dataframe(result, use_container_width=True, hide_index=True)
                    d1, d2 = st.columns(2)
                    with d1:
                        st.download_button("下载CSV结果", result.to_csv(index=False, encoding="utf-8-sig"), file_name="mp_response_predictions.csv", mime="text/csv", use_container_width=True)
                    with d2:
                        st.download_button("下载Excel结果", dataframe_to_xlsx_bytes(result), file_name="mp_response_predictions.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
                except Exception as exc:
                    st.error(f"批量预测失败：{exc}")
        with col2:
            st.markdown(
                """
                <div class="side-guide compact-guide">
                  <div class="eyebrow">Template fields</div>
                  <p>年龄、钙化、曲线、ADC值、cm、ER、PR、HER2、Ki-67</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if SAMPLE_CSV_PATH.exists():
                st.download_button("下载CSV模板", SAMPLE_CSV_PATH.read_bytes(), file_name="sample_input.csv", mime="text/csv", use_container_width=True)
            if SAMPLE_XLSX_PATH.exists():
                st.download_button("下载Excel模板", SAMPLE_XLSX_PATH.read_bytes(), file_name="sample_input.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)


def render_guide() -> None:
    st.markdown("<a id='guide'></a>", unsafe_allow_html=True)
    st.markdown("<div class='section-heading'><span>03</span><h2>Input guide and interpretation</h2></div>", unsafe_allow_html=True)
    guide = pd.DataFrame(
        [
            ["Clinical", "年龄", "输入患者年龄，单位为岁。"],
            ["MRI", "肿瘤最大径", "输入治疗前MRI或临床记录中的最大径，单位cm。"],
            ["MRI", "ADC值", "输入治疗前MRI ADC测量值。"],
            ["MRI", "钙化", "选择有/无。"],
            ["MRI", "动态增强曲线", "选择流出、平台、持续或未知。"],
            ["IHC", "ER / PR", "输入百分比，0–100。"],
            ["IHC", "HER2", "选择0、1+、2+、3+。"],
            ["IHC", "Ki-67", "输入百分比，0–100。"],
        ],
        columns=["Domain", "Feature", "How to input"],
    )
    g1, g2 = st.columns([0.64, 0.36], gap="large")
    with g1:
        st.dataframe(guide, use_container_width=True, hide_index=True)
    with g2:
        st.markdown(
            """
            <div class="research-boundary">
              <div class="eyebrow">Interpretation boundary</div>
              <h3>如何理解输出？</h3>
              <p>模型输出的是MP 4–5良好反应的概率估计，不是确定诊断。</p>
              <p>低、中、高分层用于辅助研究展示和病例分层，不能替代病理诊断、影像阅片或MDT决策。</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_footer() -> None:
    st.markdown(
        """
        <footer class="app-footer">
          <div><b>Breast Cancer MRI–IHC MP Response Feature Calculator</b></div>
          <div>Research use only · Feature input → Probability output</div>
        </footer>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    load_css()
    model, metadata = load_artifact()
    threshold = float(metadata["operating_threshold"])

    render_top_nav()
    render_hero()

    left, right = st.columns([1.12, 0.88], gap="large")
    with left:
        with st.container(border=True):
            submitted, values = render_input_form()
    with right:
        render_feature_summary(values)
        single_df = input_to_dataframe(values)
        if submitted:
            prob = float(model.predict_proba(single_df)[:, 1][0])
            render_result_card(prob, threshold, single_df)
        else:
            render_pending_card()

    render_batch_prediction(model, threshold)
    render_guide()
    render_footer()


if __name__ == "__main__":
    main()
