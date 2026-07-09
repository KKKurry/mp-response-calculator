# 乳腺癌 Miller–Payne 良好反应预测系统 V15

V15 是稳定中文看板的二分类判定版。系统输入治疗前临床、MRI 与免疫组化指标，输出是否预测为 MP 4–5 级良好病理反应。

## 版本重点

- 输出主结果改为：是否预测为 MP 4–5（是 / 否）
- 不再把模型连续输出值称为“预测概率”
- 不再显示低 / 中 / 高反应可能性分层
- 显示模型输出值、固定判定阈值和判定规则
- 增加阈值来源说明：开发集交叉验证 + Youden 指数最大化
- 保持 `app.py + assets/style.css` 分离架构
- 模型权重仅在点击预测按钮后延迟加载，降低 Streamlit 白屏风险

## 本地运行

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 核心文件

```text
app.py
model_pipeline.py
assets/style.css
model/mp_rf_model.joblib
model/metadata.json
sample_input.csv
sample_input.xlsx
```

## 部署提醒

如果你的 GitHub 仓库中已经有 `model/mp_rf_model.joblib`，更新线上版本时可以只覆盖：

```text
app.py
model_pipeline.py
assets
README.md
README_DEPLOY_FOR_BEGINNERS.md
requirements.txt
sample_input.csv
sample_input.xlsx
preview.html
```

避免反复上传模型大文件。
