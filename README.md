# 乳腺癌 MP 良好反应预测系统 · v12 温馨医学 Dashboard 版

本版本采用温馨乳腺癌医学背景、玻璃拟态卡片、中文临床 Dashboard 双栏布局和大字号预测结果看板。

## 本地运行
```bash
pip install -r requirements.txt
streamlit run app.py
```

## GitHub / Streamlit 更新建议
若原仓库已有 `model/` 文件夹，可优先上传：
- `app.py`
- `model_pipeline.py`
- `assets/style.css`
- `assets/warm_breast_background.png`
- `docs/MODEL_CARD.md`

然后在 Streamlit 中 Reboot app。
