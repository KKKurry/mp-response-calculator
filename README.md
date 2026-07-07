# 乳腺癌 MP 良好反应预测系统

这是中文 Dashboard 版网页计算器，用于基于治疗前临床特征、MRI影像特征和免疫组化指标估计 Miller–Payne 4–5 级良好病理反应概率。

## 本版视觉特点

- 纯中文医学界面
- 左侧输入表单，右侧实时病例概览与结果看板
- 大字号概率结果展示
- 渐变概率进度条
- 低 / 中 / 高反应可能性标签
- 轻量化输入组件与软阴影卡片
- 支持单病例预测与批量预测

## 本地运行

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 部署

上传本文件夹内所有可见内容到 GitHub 仓库根目录，并在 Streamlit Cloud 中部署 `app.py`。
