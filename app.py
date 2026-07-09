from __future__ import annotations

from html import escape
from io import BytesIO
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# 让 joblib 能正确加载训练时使用的自定义特征工程组件
from model_pipeline import MODEL_FEATURES  # noqa: F401

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model" / "mp_rf_model.joblib"
CSS_PATH = APP_DIR / "assets" / "style.css"
SAMPLE_CSV_PATH = APP_DIR / "sample_input.csv"
SAMPLE_XLSX_PATH = APP_DIR / "sample_input.xlsx"

st.set_page_config(
    page_title="乳腺癌MP良好反应预测系统",
    page_icon="🧬",
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


def score_text(value: float, digits: int = 3) -> str:
    """模型连续输出值。该数值仅用于阈值二分类，不解释为真实临床发生概率。"""
    return f"{float(value):.{digits}f}"


def binary_info(value: float, threshold: float) -> dict[str, str]:
    """根据固定操作阈值给出 MP 4–5 二分类判定。"""
    positive = float(value) >= float(threshold)
    if positive:
        return {
            "flag": "是",
            "class": "binary-yes",
            "group": "预测为 MP 4–5 级良好病理反应",
            "rule": f"模型输出值 ≥ {score_text(threshold)}",
            "summary": "该病例的模型输出值达到开发集中预先确定的固定二分类阈值，因此判定为 MP 4–5。",
        }
    return {
        "flag": "否",
        "class": "binary-no",
        "group": "预测为 MP 1–3 级非良好病理反应",
        "rule": f"模型输出值 < {score_text(threshold)}",
        "summary": "该病例的模型输出值未达到开发集中预先确定的固定二分类阈值，因此判定为 MP 1–3。",
    }


def input_to_dataframe(values: dict) -> pd.DataFrame:
    curve = None if values["curve"] == "未知" else values["curve"]
    return pd.DataFrame([
        {
            "年龄": values["age"],
            "钙化": values["calcification"],
            "曲线": curve,
            "ADC值": values["adc"],
            "cm": f"{values['tumor_size']:.2f}",
            "ER": f"{values['er']}%",
            "PR": f"{values['pr']}%",
            "HER2": values["her2"],
            "Ki-67": f"{values['ki67']}%",
        }
    ])


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
        df.to_excel(writer, index=False, sheet_name="预测结果")
    return output.getvalue()


def html_kv(label: str, value: str, accent: bool = False) -> str:
    cls = "case-kv accent" if accent else "case-kv"
    return f'<div class="{cls}"><span>{escape(label)}</span><b>{escape(value)}</b></div>'


def render_header() -> None:
    st.markdown(
        """
        <div class="top-ambient">
          <div class="ambient-line"></div>
          <div class="ambient-dots">
            <span>MRI</span><span>ADC</span><span>IHC</span><span>MP 4–5</span><span>疗效预测</span>
          </div>
        </div>
        <header class="cn-nav">
          <div class="brand-wrap">
            <div class="brand-mark">MP</div>
            <div class="brand-text"><strong>乳腺癌疗效预测系统</strong><span>治疗前多模态特征智能计算</span></div>
          </div>
          <nav>
            <a href="#calculator">单病例预测</a>
            <a href="#batch">批量预测</a>
            <a href="#guide">模型说明</a>
            <a href="#boundary">使用边界</a>
          </nav>
        </header>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <section class="cn-hero">
          <div>
            <span class="system-chip">医学研究型预测计算器</span>
            <h1>乳腺癌 Miller–Payne 良好反应预测系统</h1>
            <p>基于治疗前临床特征、MRI影像特征及免疫组化指标，进行 MP 4–5 级良好病理反应二分类判定。</p>
            <div class="hero-badges">
              <span><i>🎯</i><b>终点事件</b>MP 4–5</span>
              <span><i>🧲</i><b>输入维度</b>MRI / IHC / 临床特征</span>
              <span><i>📊</i><b>输出形式</b>二分类判定</span>
              <span><i>🔬</i><b>使用场景</b>科研辅助</span>
            </div>
          </div>
          <div class="hero-panel">
            <div class="hero-panel-title">计算流程</div>
            <div class="hero-step"><span>1</span><b>录入治疗前特征</b><em>临床、MRI、免疫组化</em></div>
            <div class="hero-step"><span>2</span><b>获得模型输出值</b><em>仅用于固定阈值比较</em></div>
            <div class="hero-step"><span>3</span><b>输出二分类结果</b><em>是 / 否预测为 MP 4–5</em></div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_case_preview(values: dict) -> None:
    st.markdown(
        f"""
        <div class="dashboard-card case-card">
          <div class="card-kicker">实时同步</div>
          <h3>当前病例概览</h3>
          <p>左侧参数调整后，本区域即时更新。</p>
          <div class="case-highlight-grid">
            {html_kv('年龄', f'{values["age"]}岁', True)}
            {html_kv('肿瘤最大径', f'{values["tumor_size"]:.1f} cm', True)}
            {html_kv('ADC值', f'{values["adc"]:.3f}', True)}
            {html_kv('Ki-67指数', f'{values["ki67"]}%', True)}
          </div>
          <div class="case-list">
            {html_kv('钙化情况', values['calcification'])}
            {html_kv('动态增强曲线', values['curve'])}
            {html_kv('ER表达比例', f'{values["er"]}%')}
            {html_kv('PR表达比例', f'{values["pr"]}%')}
            {html_kv('HER2评分', values['her2'])}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_result(stale: bool = False) -> None:
    title = "参数已更新" if stale else "尚未计算"
    msg = "当前输入已发生变化，请重新点击开始判定。" if stale else "请确认左侧治疗前特征，点击开始判定后生成 MP 4–5 二分类结果。"
    st.markdown(
        f"""
        <div class="dashboard-card result-card empty-state">
          <div class="card-kicker">二分类判定</div>
          <h3>{title}</h3>
          <div class="empty-percent">--</div>
          <div class="skeleton-bar"><span></span></div>
          <div class="empty-tag">等待判定</div>
          <p>{msg}</p>
          <div class="skeleton-lines"><i></i><i></i><i></i></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result(model_output: float, threshold: float) -> None:
    info = binary_info(model_output, threshold)
    st.markdown(
        f"""
        <div class="dashboard-card result-card binary-result {info['class']}">
          <div class="card-kicker">模型二分类判定结果</div>
          <h3>是否预测为 MP 4–5 级良好反应</h3>
          <div class="binary-answer">{info['flag']}</div>
          <div class="binary-tag">{info['group']}</div>
          <div class="threshold-grid">
            <div><span>模型输出值</span><b>{score_text(model_output)}</b></div>
            <div><span>固定判定阈值</span><b>{score_text(threshold)}</b></div>
          </div>
          <div class="model-group"><span>判定规则</span><b>{info['rule']}</b></div>
          <p class="result-summary">{info['summary']}</p>
          <p class="result-advice">模型输出值仅用于与固定阈值比较，不应解释为患者达到 MP 4–5 级的真实临床发生概率。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_interpretation(threshold: float) -> None:
    st.markdown(
        f"""
        <div class="dashboard-card interpretation-card">
          <div class="card-kicker">阈值说明</div>
          <h3>二分类阈值如何确定</h3>
          <div class="interpret-row high"><b>阈值来源</b><span>仅在开发集交叉验证输出中确定，后续测试集和网页预测阶段保持固定。</span></div>
          <div class="interpret-row mid"><b>确定方法</b><span>采用 Youden 指数最大化原则，即选择 sensitivity + specificity − 1 最大的截断点。</span></div>
          <div class="interpret-row low"><b>当前阈值</b><span>固定判定阈值为 {score_text(threshold)}。模型输出值达到或超过该值时判定为 MP 4–5，否则判定为 MP 1–3。</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_feature_docs() -> None:
    st.markdown(
        """
        <section class="doc-section" id="guide">
          <div class="section-title">
            <span>系统说明</span>
            <h2>变量、输出与使用边界</h2>
            <p>详细解释默认收纳在说明卡片中，避免干扰单病例预测主流程。</p>
          </div>
          <div class="doc-grid">
            <div class="doc-card"><b>输入变量</b><p>模型使用年龄、肿瘤最大径、ADC值、钙化情况、动态增强曲线、ER、PR、HER2及Ki-67等治疗前可获得特征。</p></div>
            <div class="doc-card"><b>输出结果</b><p>系统输出模型连续输出值、固定判定阈值及是否预测为 MP 4–5 的二分类结果。</p></div>
            <div class="doc-card"><b>结果解释</b><p>模型输出值仅用于阈值比较，不应解释为真实临床发生概率；最终结论以二分类判定为准。</p></div>
            <div class="doc-card" id="boundary"><b>使用边界</b><p>本系统仅供科研展示和模型转化原型使用，不替代临床诊断、治疗决策或多学科会诊意见。</p></div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    load_css()
    model, metadata = load_artifact()
    threshold = float(metadata.get("operating_threshold", 0.5))

    render_header()
    render_hero()

    st.markdown('<div class="section-title" id="calculator"><span>核心功能</span><h2>单病例预测看板</h2><p>左侧输入治疗前特征，右侧实时同步病例概览并显示是否预测为 MP 4–5 的二分类结果。</p></div>', unsafe_allow_html=True)

    left, right = st.columns([1.32, 0.88], gap="large")

    with left:
        st.markdown('<div class="input-panel"><div class="panel-title"><span>01</span><div><b>单病例特征输入</b><em>请输入治疗前可获得指标</em></div></div>', unsafe_allow_html=True)

        st.markdown('<div class="input-group-title">临床与病灶特征 <span title="治疗前可获得的基础临床信息及病灶负荷指标">ⓘ</span></div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("年龄", min_value=18, max_value=100, value=52, step=1, help="患者治疗前年龄，单位：岁。")
        with c2:
            tumor_size = st.number_input("肿瘤最大径", min_value=0.1, max_value=15.0, value=2.8, step=0.1, format="%.1f", help="治疗前影像或病理记录中的病灶最大径，单位：cm。")
        with c3:
            adc = st.number_input("ADC值", min_value=0.100, max_value=3.000, value=0.950, step=0.001, format="%.3f", help="治疗前MRI弥散加权成像获得的ADC值。")

        st.markdown('<div class="input-group-title">治疗前 MRI 特征 <span title="治疗前影像评估中记录的关键表型特征">ⓘ</span></div>', unsafe_allow_html=True)
        m1, m2 = st.columns([0.82, 1.18])
        with m1:
            calcification = st.radio("钙化情况", ["有钙化", "无钙化"], horizontal=True, index=0)
        with m2:
            curve = st.selectbox("动态增强曲线类型", ["流出型", "平台型", "持续型", "未知"], index=0, help="请选择治疗前DCE-MRI动态增强曲线类型。")

        st.markdown('<div class="input-group-title">免疫组化指标 <span title="治疗前穿刺或病理检查获得的免疫组化表达信息">ⓘ</span></div>', unsafe_allow_html=True)
        i1, i2, i3, i4 = st.columns(4)
        with i1:
            er = st.number_input("ER表达比例", min_value=0, max_value=100, value=10, step=1, help="单位：%。")
        with i2:
            pr = st.number_input("PR表达比例", min_value=0, max_value=100, value=5, step=1, help="单位：%。")
        with i3:
            her2 = st.selectbox("HER2评分", ["0", "1+", "2+", "3+"], index=0)
        with i4:
            ki67 = st.number_input("Ki-67指数", min_value=0, max_value=100, value=70, step=1, help="单位：%。")

        values = {
            "age": int(age),
            "tumor_size": float(tumor_size),
            "adc": float(adc),
            "calcification": calcification,
            "curve": curve,
            "er": int(er),
            "pr": int(pr),
            "her2": her2,
            "ki67": int(ki67),
        }
        signature = tuple(values.items())

        st.markdown('<div class="input-note">提示：所有输入均应来自治疗前资料。点击“开始计算”后，系统将在右侧生成模型输出值与 MP 4–5 二分类判定结果。</div>', unsafe_allow_html=True)
        calculate_clicked = st.button("开始判定是否为 MP 4–5", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if calculate_clicked:
            one_df = input_to_dataframe(values)
            prob = float(model.predict_proba(one_df)[:, 1][0])
            st.session_state["last_model_output"] = prob
            st.session_state["last_signature"] = signature
            st.session_state["last_input_df"] = one_df

    with right:
        st.markdown('<div class="right-sticky">', unsafe_allow_html=True)
        render_case_preview(values)
        last_prob = st.session_state.get("last_model_output")
        last_sig = st.session_state.get("last_signature")
        if last_prob is None:
            render_empty_result(stale=False)
        elif last_sig != signature:
            render_empty_result(stale=True)
        else:
            render_result(float(last_prob), threshold)
            result_df = st.session_state.get("last_input_df", input_to_dataframe(values)).copy()
            decision = binary_info(float(last_prob), threshold)
            result_df["模型输出值"] = score_text(float(last_prob))
            result_df["固定判定阈值"] = score_text(threshold)
            result_df["是否预测为MP4-5"] = decision["flag"]
            result_df["预测MP分组"] = decision["group"]
            result_df["判定规则"] = decision["rule"]
            st.download_button(
                "下载单病例结果",
                data=result_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="single_case_prediction.csv",
                mime="text/csv",
                use_container_width=True,
            )
        render_interpretation(threshold)
        st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("批量预测 / 上传CSV或Excel文件", expanded=False):
        st.markdown("请上传包含模型输入变量的CSV或Excel文件。系统将自动追加模型输出值、固定阈值与是否预测为 MP 4–5 的二分类结果。")
        uploaded = st.file_uploader("上传批量数据文件", type=["csv", "xlsx", "xls"])
        dl1, dl2 = st.columns(2)
        if SAMPLE_CSV_PATH.exists():
            dl1.download_button("下载CSV模板", data=SAMPLE_CSV_PATH.read_bytes(), file_name="sample_input.csv", mime="text/csv", use_container_width=True)
        if SAMPLE_XLSX_PATH.exists():
            dl2.download_button("下载Excel模板", data=SAMPLE_XLSX_PATH.read_bytes(), file_name="sample_input.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

        if uploaded is not None:
            try:
                raw = read_uploaded_table(uploaded)
                batch = normalize_batch_columns(raw)
                probs = model.predict_proba(batch)[:, 1]
                out = raw.copy()
                decisions = [binary_info(float(p), threshold) for p in probs]
                out["模型输出值"] = [score_text(float(p)) for p in probs]
                out["固定判定阈值"] = score_text(threshold)
                out["是否预测为MP4-5"] = [d["flag"] for d in decisions]
                out["预测MP分组"] = [d["group"] for d in decisions]
                out["判定规则"] = [d["rule"] for d in decisions]
                st.dataframe(out, use_container_width=True, height=320)
                b1, b2 = st.columns(2)
                b1.download_button("下载CSV结果", data=out.to_csv(index=False).encode("utf-8-sig"), file_name="batch_predictions.csv", mime="text/csv", use_container_width=True)
                b2.download_button("下载Excel结果", data=dataframe_to_xlsx_bytes(out), file_name="batch_predictions.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
            except Exception as exc:  # noqa: BLE001
                st.error(f"批量预测失败：{exc}")

    render_feature_docs()
    st.markdown('<footer class="cn-footer">本系统仅供科研和模型展示使用，不替代临床诊断、治疗决策或多学科会诊意见。</footer>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
