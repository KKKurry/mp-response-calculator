# 乳腺癌 Miller–Payne 良好反应预测系统 v14

这是一个基于 Streamlit 的中文医学科研辅助看板，用于根据治疗前临床特征、MRI 影像特征及免疫组化指标，预测新辅助治疗后达到 Miller–Payne 4–5 级良好病理反应的个体化概率。

## v14 稳定架构

- `app.py`：仅负责 Streamlit 原生组件、状态管理、数据处理、延迟模型加载与推理。
- `assets/style.css`：负责页面背景、卡片、输入框、按钮、结果大字号、进度条等所有视觉样式。
- 模型权重仅在点击“🚀 开始精准预测”或执行批量预测时加载，避免页面初始化白屏。
- 不使用复杂多层嵌套 HTML，不将 Streamlit 原生交互组件嵌入自定义 HTML 容器。

## 本地运行

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 上传到 GitHub Desktop

推荐只覆盖以下文件和文件夹：

```text
app.py
model_pipeline.py
assets
docs
README.md
README_DEPLOY_FOR_BEGINNERS.md
requirements.txt
sample_input.csv
sample_input.xlsx
preview.html
```

如果你的远程仓库已经有 `model/mp_rf_model.joblib`，可以不重复上传 `model` 文件夹，避免大文件上传失败。

## 研究用途声明

本系统仅供医学科研辅助、模型展示和论文补充材料原型使用，不替代临床诊断、治疗决策或多学科会诊意见。
