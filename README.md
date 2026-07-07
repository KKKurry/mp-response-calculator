# 乳腺癌 MP 良好反应预测系统｜v12 Fixed Light

该版本为 v12 温馨医学 Dashboard 的轻量修复版：

- 保留纯中文医学界面
- 保留温馨粉蓝医疗风格
- 保留左侧输入、右侧结果看板
- 保留大字号预测概率、渐变进度条和反应可能性标签
- 移除大体积 base64 背景图嵌入，避免 Streamlit 前端白屏或卡死
- 使用轻量 CSS 渐变和玻璃卡片提升稳定性

部署时建议只覆盖：`app.py`、`assets/style.css`、`model_pipeline.py`、`docs/`、`README.md`、`requirements.txt`。
如果原仓库的 `model/` 文件夹仍在，可暂时不上传模型文件夹。
