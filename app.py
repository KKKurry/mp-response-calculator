from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st

# 保留轻量导入，保证 joblib 反序列化时可以找到自定义特征工程类。
# 模型权重本身只在用户点击预测按钮后加载，避免初始化阶段白屏。
from model_pipeline import MODEL_FEATURES  # noqa: F401

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model" / "mp_rf_model.joblib"
CSS_PATH = APP_DIR / "assets" / "style.css"
SAMPLE_CSV_PATH = APP_DIR / "sample_input.csv"
SAMPLE_XLSX_PATH = APP_DIR / "sample_input.xlsx"

st.set_page_config(
    page_title="乳腺癌MP二分类预测系统",
    page_icon="🎀",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================
# 基础工具函数
# =========================

def load_css() -> None:
    """加载独立 CSS 文件。所有视觉样式均维护在 assets/style.css 中。"""
    if CSS_PATH.exists():
        st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_model_artifact() -> tuple[Any, dict[str, Any]]:
    """延迟加载模型：仅在用户点击预测按钮或批量预测按钮后调用。"""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"未找到模型文件：{MODEL_PATH}")
    artifact = joblib.load(MODEL_PATH)
    return artifact["model"], artifact.get("metadata", {})


def get_threshold(metadata: dict[str, Any]) -> float:
    """读取固定二分类判定阈值；若模型元数据缺失则使用 0.5 作为兜底。"""
    return float(metadata.get("operating_threshold", 0.5))


def format_score(score: float) -> str:
    """模型连续输出值格式化。注意：不把该值解释为真实临床发生概率。"""
    return f"{float(score):.3f}"


def classify_by_threshold(score: float, threshold: float) -> dict[str, str]:
    """基于固定阈值进行 MP 4–5 二分类判定。"""
    positive = float(score) >= float(threshold)
    if positive:
        return {
            "binary": "是",
            "css": "binary-yes",
            "group": "预测为 MP 4–5 级良好病理反应",
            "short_group": "MP 4–5",
            "rule": "模型输出值 ≥ 固定判定阈值",
            "summary": "该病例的模型输出值达到预设二分类阈值，系统判定为 MP 4–5 级良好病理反应组。",
        }
    return {
        "binary": "否",
        "css": "binary-no",
        "group": "预测为 MP 1–3 级非良好病理反应",
        "short_group": "MP 1–3",
        "rule": "模型输出值 < 固定判定阈值",
        "summary": "该病例的模型输出值未达到预设二分类阈值，系统判定为 MP 1–3 级非良好病理反应组。",
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
    return [col for col in required if col not in df.columns]


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
        df.to_excel(writer, index=False, sheet_name="二分类预测结果")
    return output.getvalue()


# =========================
# 页面模块
# =========================

def render_header() -> None:
    st.markdown('<div class="top-accent"></div>', unsafe_allow_html=True)
    left, right = st.columns([1.15, 1.0], vertical_alignment="center")
    with left:
        st.markdown('<p class="system-label">医学科研辅助看板 · 二分类判定版</p>', unsafe_allow_html=True)
        st.title("乳腺癌 Miller–Payne 良好反应预测系统")
        st.caption("基于治疗前临床特征、MRI影像特征及免疫组化指标，输出是否预测为 MP 4–5 级良好病理反应。")
    with right:
        c1, c2, c3 = st.columns(3)
        c1.metric("终点事件", "MP 4–5")
        c2.metric("判定方式", "二分类")
        c3.metric("阈值来源", "开发集")
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


def render_result_panel(score: float | None, threshold: float, stale: bool) -> None:
    st.subheader("模型二分类判定结果")

    if score is None:
        st.markdown('<p class="binary-result binary-empty">待判定</p>', unsafe_allow_html=True)
        st.info("尚未计算。请确认左侧治疗前特征，点击“🚀 开始精准预测”后生成二分类判定结果。", icon="⏳")
        st.caption("系统不会将模型输出值解释为患者真实临床发生概率。")
        return

    if stale:
        st.markdown('<p class="binary-result binary-empty">待更新</p>', unsafe_allow_html=True)
        st.warning("当前输入已发生变化，请重新点击“🚀 开始精准预测”以更新判定结果。", icon="⚠️")
        return

    result = classify_by_threshold(score, threshold)
    st.markdown(f'<p class="binary-result {result["css"]}">{result["binary"]}</p>', unsafe_allow_html=True)
    st.markdown(f'<span class="binary-tag {result["css"]}">{result["group"]}</span>', unsafe_allow_html=True)

    m1, m2 = st.columns(2)
    m1.metric("模型输出值", format_score(score))
    m2.metric("固定判定阈值", format_score(threshold))

    st.info(f"判定规则：**{result['rule']}**。", icon="📌")
    st.caption("模型输出值仅用于与固定阈值比较，不应直接解释为患者达到 MP 4–5 级的真实临床概率。")
    st.success(result["summary"])


def single_case_calculator() -> None:
    left, right = st.columns([1.22, 0.98], gap="large")

    with left:
        values = collect_inputs()
        signature = tuple(values.items())
        if st.button("🚀 开始精准预测", type="primary", use_container_width=True):
            try:
                with st.spinner("模型正在加载并计算，请稍候……"):
                    model, metadata = load_model_artifact()
                    threshold = get_threshold(metadata)
                    input_df = values_to_dataframe(values)
                    score = float(model.predict_proba(input_df)[:, 1][0])
                st.session_state["last_score"] = score
                st.session_state["last_threshold"] = threshold
                st.session_state["last_signature"] = signature
                st.session_state["last_input_df"] = input_df
            except Exception as exc:  # noqa: BLE001
                st.error(f"模型加载或预测失败：{exc}")

    with right:
        with st.container(border=True):
            render_input_overview(values)
        with st.container(border=True):
            last_score = st.session_state.get("last_score")
            last_threshold = float(st.session_state.get("last_threshold", 0.5))
            last_signature = st.session_state.get("last_signature")
            stale = last_score is not None and last_signature != signature
            render_result_panel(None if last_score is None else float(last_score), last_threshold, stale)

            if last_score is not None and not stale:
                result = classify_by_threshold(float(last_score), last_threshold)
                result_df = st.session_state.get("last_input_df", values_to_dataframe(values)).copy()
                result_df["模型输出值"] = format_score(float(last_score))
                result_df["固定判定阈值"] = format_score(last_threshold)
                result_df["是否预测为MP4-5"] = result["binary"]
                result_df["预测MP分组"] = result["short_group"]
                result_df["判定规则"] = result["rule"]
                st.download_button(
                    "下载本例判定结果",
                    data=result_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="single_case_mp_binary_result.csv",
                    mime="text/csv",
                    use_container_width=True,
                )


def batch_prediction() -> None:
    with st.expander("批量预测 / 上传 CSV 或 Excel", expanded=False):
        st.caption("批量结果输出为二分类判定结果；模型输出值仅用于阈值比较，不解释为真实临床概率。")
        uploaded_file = st.file_uploader("上传批量病例表", type=["csv", "xlsx", "xls"])

        c1, c2 = st.columns(2)
        if SAMPLE_CSV_PATH.exists():
            c1.download_button(
                "下载CSV模板",
                data=SAMPLE_CSV_PATH.read_bytes(),
                file_name="sample_input.csv",
                mime="text/csv",
                use_container_width=True,
            )
        if SAMPLE_XLSX_PATH.exists():
            c2.download_button(
                "下载Excel模板",
                data=SAMPLE_XLSX_PATH.read_bytes(),
                file_name="sample_input.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

        if uploaded_file is not None and st.button("开始批量二分类判定", use_container_width=True):
            try:
                raw = read_uploaded_table(uploaded_file)
                df = normalize_batch_columns(raw)
                missing = validate_batch_table(df)
                if missing:
                    st.error("上传文件缺少必要字段：" + "、".join(missing))
                    return

                with st.spinner("模型正在加载并进行批量判定……"):
                    model, metadata = load_model_artifact()
                    threshold = get_threshold(metadata)
                    scores = model.predict_proba(df)[:, 1]

                out = raw.copy()
                out["模型输出值"] = [format_score(float(x)) for x in scores]
                out["固定判定阈值"] = format_score(threshold)
                out["是否预测为MP4-5"] = [classify_by_threshold(float(x), threshold)["binary"] for x in scores]
                out["预测MP分组"] = [classify_by_threshold(float(x), threshold)["short_group"] for x in scores]
                out["判定规则"] = [classify_by_threshold(float(x), threshold)["rule"] for x in scores]

                st.success(f"批量二分类判定完成，共 {len(out)} 条记录。")
                st.dataframe(out, use_container_width=True, hide_index=True)

                dl1, dl2 = st.columns(2)
                dl1.download_button(
                    "下载CSV结果",
                    data=out.to_csv(index=False).encode("utf-8-sig"),
                    file_name="mp_binary_prediction_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
                dl2.download_button(
                    "下载Excel结果",
                    data=dataframe_to_xlsx_bytes(out),
                    file_name="mp_binary_prediction_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"批量预测失败：{exc}")


def documentation() -> None:
    st.markdown("---")
    st.subheader("模型判定规则与使用说明")

    d1, d2 = st.columns(2)
    with d1:
        with st.container(border=True):
            st.markdown("##### 二分类输出")
            st.write("本系统的前端主结果为：**是否预测为 MP 4–5 级良好病理反应**。")
            st.write("输出结果分为：**是，预测为 MP 4–5**；或 **否，预测为 MP 1–3**。")
    with d2:
        with st.container(border=True):
            st.markdown("##### 非概率解释声明")
            st.write("网页中的模型输出值仅用于与固定阈值比较，不应解释为患者真实临床发生概率。")
            st.write("因此，本系统不再展示“高/中/低反应可能性”作为主结论。")

    with st.expander("阈值如何确定？", expanded=True):
        st.write(
            "模型先输出连续的模型输出值。为了将连续输出值转化为 MP 4–5 与 MP 1–3 的二分类结果，"
            "本系统使用开发集中预先确定的固定操作阈值。"
        )
        st.write(
            "该阈值基于开发集交叉验证输出，并采用 Youden 指数最大化原则确定，即选择使 "
            "sensitivity + specificity − 1 最大的截断点。阈值确定后保持固定，不在测试集或在线预测阶段重新调整。"
        )
        st.info(
            "判定规则：模型输出值 ≥ 固定阈值 → 预测为 MP 4–5；模型输出值 < 固定阈值 → 预测为 MP 1–3。",
            icon="📌",
        )

    with st.expander("输入变量说明", expanded=False):
        st.write("输入变量均为治疗前可获得特征，包括临床与病灶特征、治疗前 MRI 特征及免疫组化指标。")
        st.write("变量包括：年龄、肿瘤最大径、ADC值、钙化情况、动态增强曲线类型、ER、PR、HER2、Ki-67。")

    with st.expander("研究用途声明", expanded=False):
        st.write("本系统为医学科研辅助看板和模型转化展示原型，仅供研究使用。")
        st.write("系统输出不构成诊断意见，也不能替代多学科临床决策。")


def main() -> None:
    load_css()
    render_header()
    single_case_calculator()
    batch_prediction()
    documentation()


if __name__ == "__main__":
    main()
