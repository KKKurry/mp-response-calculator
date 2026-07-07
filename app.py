from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st

# 仅保留轻量模块导入，保证 joblib 反序列化时可找到自定义转换器。
# 模型权重本身不会在页面初始化阶段加载。
from model_pipeline import MODEL_FEATURES  # noqa: F401

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model" / "mp_rf_model.joblib"
CSS_PATH = APP_DIR / "assets" / "style.css"
SAMPLE_CSV_PATH = APP_DIR / "sample_input.csv"
SAMPLE_XLSX_PATH = APP_DIR / "sample_input.xlsx"

st.set_page_config(
    page_title="乳腺癌MP良好反应预测系统",
    page_icon="🎀",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def load_css() -> None:
    """加载独立样式文件。所有视觉表现均维护在 style.css 中。"""
    if CSS_PATH.exists():
        st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_model_artifact() -> tuple[Any, dict[str, Any]]:
    """延迟加载模型；只在用户点击预测或批量预测时调用。"""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"未找到模型文件：{MODEL_PATH}")
    artifact = joblib.load(MODEL_PATH)
    return artifact["model"], artifact.get("metadata", {})


def format_probability(prob: float, digits: int = 1) -> str:
    return f"{float(prob) * 100:.{digits}f}%"


def tier_info(prob: float) -> dict[str, str]:
    """将个体化概率映射为科研展示用反应可能性分层。"""
    if prob < 0.40:
        return {
            "label": "低反应可能性",
            "css": "tier-low",
            "summary": "模型提示该病例达到 MP 4–5 级良好病理反应的可能性较低，建议结合影像、病理和临床治疗方案综合判断。",
        }
    if prob < 0.70:
        return {
            "label": "中等反应可能性",
            "css": "tier-mid",
            "summary": "模型提示该病例处于中等反应可能性区间，单一模型结果不宜作为治疗决策依据。",
        }
    return {
        "label": "高反应可能性",
        "css": "tier-high",
        "summary": "模型提示该病例达到 MP 4–5 级良好病理反应的可能性较高，具有较好的治疗反应倾向。",
    }


def values_to_dataframe(values: dict[str, Any]) -> pd.DataFrame:
    curve_map = {
        "流出型": "流出",
        "平台型": "平台",
        "持续型": "持续",
        "未知": None,
    }
    return pd.DataFrame(
        [
            {
                "年龄": values["age"],
                "钙化": "有" if values["calcification"] == "有钙化" else "无",
                "曲线": curve_map.get(values["curve"], values["curve"]),
                "ADC值": values["adc"],
                "cm": f"{values['tumor_size']:.2f}",
                "ER": f"{values['er']}%",
                "PR": f"{values['pr']}%",
                "HER2": values["her2"],
                "Ki-67": f"{values['ki67']}%",
            }
        ]
    )


def normalize_batch_columns(df: pd.DataFrame) -> pd.DataFrame:
    """兼容中文字段与常见英文字段。"""
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


def validate_batch_table(df: pd.DataFrame) -> list[str]:
    required = ["年龄", "钙化", "曲线", "ADC值", "cm", "ER", "PR", "HER2", "Ki-67"]
    missing = [col for col in required if col not in df.columns]
    return missing


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


def render_header() -> None:
    st.markdown('<div class="top-accent"></div>', unsafe_allow_html=True)
    top_left, top_right = st.columns([1.1, 1.0], vertical_alignment="center")
    with top_left:
        st.markdown('<p class="system-label">医学科研辅助看板</p>', unsafe_allow_html=True)
        st.title("乳腺癌 Miller–Payne 良好反应预测系统")
        st.caption("基于治疗前临床特征、MRI影像特征及免疫组化指标，评估 MP 4–5 级良好病理反应概率。")
    with top_right:
        b1, b2, b3 = st.columns(3)
        b1.metric("终点事件", "MP 4–5")
        b2.metric("输入维度", "9项特征")
        b3.metric("输出形式", "个体概率")
    st.markdown("---")


def collect_inputs() -> dict[str, Any]:
    st.subheader("单病例特征输入")
    st.caption("请输入治疗前可获得指标；右侧病例概览将实时同步更新。")

    with st.container(border=True):
        st.markdown("##### 临床与病灶特征")
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("年龄", min_value=18, max_value=100, value=52, step=1, help="单位：岁。")
        tumor_size = c2.number_input("肿瘤最大径", min_value=0.1, max_value=15.0, value=2.8, step=0.1, format="%.1f", help="单位：cm。")
        adc = c3.number_input("ADC值", min_value=0.100, max_value=3.000, value=0.950, step=0.001, format="%.3f", help="治疗前MRI弥散加权成像相关测量值。")

    with st.container(border=True):
        st.markdown("##### 治疗前 MRI 特征")
        m1, m2 = st.columns([0.85, 1.15])
        calcification = m1.radio("钙化情况", ["有钙化", "无钙化"], horizontal=True)
        curve = m2.selectbox("动态增强曲线类型", ["流出型", "平台型", "持续型", "未知"])

    with st.container(border=True):
        st.markdown("##### 免疫组化指标")
        i1, i2, i3, i4 = st.columns(4)
        er = i1.number_input("ER表达比例", min_value=0, max_value=100, value=10, step=1, help="单位：%。")
        pr = i2.number_input("PR表达比例", min_value=0, max_value=100, value=5, step=1, help="单位：%。")
        her2 = i3.selectbox("HER2评分", ["0", "1+", "2+", "3+"])
        ki67 = i4.number_input("Ki-67指数", min_value=0, max_value=100, value=70, step=1, help="单位：%。")

    return {
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


def render_input_overview(values: dict[str, Any]) -> None:
    st.subheader("当前病例概览")
    st.caption("左侧输入参数已实时同步。")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("年龄", f"{values['age']}岁")
        st.metric("ADC值", f"{values['adc']:.3f}")
        st.metric("ER表达比例", f"{values['er']}%")
        st.metric("HER2评分", values["her2"])
    with c2:
        st.metric("肿瘤最大径", f"{values['tumor_size']:.1f} cm")
        st.metric("钙化情况", values["calcification"])
        st.metric("PR表达比例", f"{values['pr']}%")
        st.metric("Ki-67指数", f"{values['ki67']}%")
    st.info(f"动态增强曲线类型：**{values['curve']}**", icon="🧭")


def render_result_panel(probability: float | None, threshold: float, stale: bool) -> None:
    st.subheader("预测结果")
    if probability is None:
        st.markdown('<p class="result-number empty-result">--.-%</p>', unsafe_allow_html=True)
        st.progress(0.30)
        st.info("尚未计算。请确认左侧治疗前特征，点击“🚀 开始精准预测”后生成个体化预测结果。", icon="⏳")
        return

    if stale:
        st.markdown('<p class="result-number empty-result">--.-%</p>', unsafe_allow_html=True)
        st.progress(0.30)
        st.warning("当前输入已发生变化，请重新点击“🚀 开始精准预测”以更新结果。", icon="⚠️")
        return

    tier = tier_info(probability)
    predicted_group = "倾向 MP 4–5 良好反应" if probability >= threshold else "倾向 MP 1–3 非良好反应"

    st.markdown(f'<p class="result-number">{format_probability(probability)}</p>', unsafe_allow_html=True)
    st.progress(min(max(float(probability), 0.0), 1.0))
    st.markdown(f'<span class="tier-tag {tier["css"]}">{tier["label"]}</span>', unsafe_allow_html=True)
    st.success(f"模型预测组别：**{predicted_group}**")
    st.caption(tier["summary"])


def single_case_calculator() -> None:
    left, right = st.columns([1.25, 0.95], gap="large")

    with left:
        values = collect_inputs()
        signature = tuple(values.items())
        if st.button("🚀 开始精准预测", type="primary", use_container_width=True):
            try:
                with st.spinner("模型正在加载并计算，请稍候……"):
                    model, metadata = load_model_artifact()
                    threshold = float(metadata.get("operating_threshold", 0.5))
                    input_df = values_to_dataframe(values)
                    probability = float(model.predict_proba(input_df)[:, 1][0])
                st.session_state["last_probability"] = probability
                st.session_state["last_threshold"] = threshold
                st.session_state["last_signature"] = signature
                st.session_state["last_input_df"] = input_df
            except Exception as exc:  # noqa: BLE001
                st.error(f"模型加载或预测失败：{exc}")

    with right:
        with st.container(border=True):
            render_input_overview(values)
        with st.container(border=True):
            last_probability = st.session_state.get("last_probability")
            last_threshold = float(st.session_state.get("last_threshold", 0.5))
            last_signature = st.session_state.get("last_signature")
            stale = last_probability is not None and last_signature != signature
            render_result_panel(None if last_probability is None else float(last_probability), last_threshold, stale)

            if last_probability is not None and not stale:
                result_df = st.session_state.get("last_input_df", values_to_dataframe(values)).copy()
                result_df["MP 4–5预测概率"] = format_probability(float(last_probability))
                result_df["反应可能性分层"] = tier_info(float(last_probability))["label"]
                st.download_button(
                    "下载单病例结果",
                    data=result_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="single_case_prediction.csv",
                    mime="text/csv",
                    use_container_width=True,
                )


def batch_prediction() -> None:
    with st.expander("批量预测：上传 CSV 或 Excel 文件", expanded=False):
        st.caption("建议先下载模板，并保持字段名称一致。")
        t1, t2 = st.columns(2)
        if SAMPLE_CSV_PATH.exists():
            t1.download_button("下载CSV模板", data=SAMPLE_CSV_PATH.read_bytes(), file_name="sample_input.csv", mime="text/csv", use_container_width=True)
        if SAMPLE_XLSX_PATH.exists():
            t2.download_button("下载Excel模板", data=SAMPLE_XLSX_PATH.read_bytes(), file_name="sample_input.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

        uploaded = st.file_uploader("上传待预测数据", type=["csv", "xlsx", "xls"])
        run_batch = st.button("批量生成预测结果", use_container_width=True)

        if uploaded is not None and run_batch:
            try:
                raw = read_uploaded_table(uploaded)
                batch = normalize_batch_columns(raw)
                missing = validate_batch_table(batch)
                if missing:
                    st.error("上传文件缺少必要字段：" + "、".join(missing))
                    return

                with st.spinner("正在加载模型并进行批量预测……"):
                    model, metadata = load_model_artifact()
                    threshold = float(metadata.get("operating_threshold", 0.5))
                    probabilities = model.predict_proba(batch)[:, 1]

                output = raw.copy()
                output["MP 4–5预测概率"] = [format_probability(float(p)) for p in probabilities]
                output["反应可能性分层"] = [tier_info(float(p))["label"] for p in probabilities]
                output["模型预测组别"] = ["倾向 MP 4–5 良好反应" if float(p) >= threshold else "倾向 MP 1–3 非良好反应" for p in probabilities]

                st.dataframe(output, use_container_width=True, height=320)
                d1, d2 = st.columns(2)
                d1.download_button("下载CSV结果", data=output.to_csv(index=False).encode("utf-8-sig"), file_name="batch_predictions.csv", mime="text/csv", use_container_width=True)
                d2.download_button("下载Excel结果", data=dataframe_to_xlsx_bytes(output), file_name="batch_predictions.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
            except Exception as exc:  # noqa: BLE001
                st.error(f"批量预测失败：{exc}")


def documentation() -> None:
    st.markdown("---")
    st.subheader("模型说明")
    g1, g2, g3, g4 = st.columns(4)
    g1.info("**输入变量**\n\n年龄、肿瘤最大径、ADC值、钙化、动态增强曲线、ER、PR、HER2、Ki-67。")
    g2.info("**输出结果**\n\nMP 4–5 良好反应概率、反应可能性分层和模型预测组别。")
    g3.info("**结果解释**\n\n概率越高，提示达到良好病理反应的可能性越大。")
    g4.info("**使用边界**\n\n仅供科研辅助和模型展示，不替代临床诊断与治疗决策。")


def main() -> None:
    load_css()
    render_header()
    single_case_calculator()
    batch_prediction()
    documentation()


if __name__ == "__main__":
    main()
