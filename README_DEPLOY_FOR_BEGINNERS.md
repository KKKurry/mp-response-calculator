# 新手部署说明

## 使用 GitHub Desktop 更新网站

1. 下载并解压 v14 文件夹。
2. 在 GitHub Desktop 中打开 `mp-response-calculator` 本地仓库。
3. 点击 `Show in Explorer` 打开本地仓库目录。
4. 将 v14 文件夹中的以下内容复制进去并选择替换：

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

5. 暂时不要复制 `model` 文件夹，除非你的仓库缺少模型文件。
6. 回到 GitHub Desktop，填写 Summary：

```text
Update v14 stable Chinese dashboard
```

7. 点击 `Commit to main`。
8. 点击 `Push origin`。
9. 回到 Streamlit Cloud，点击 `Manage app → Reboot app`。
10. 等待 1–3 分钟后刷新网页。

## 需要确认的文件

仓库根目录应包含：

```text
app.py
model_pipeline.py
requirements.txt
assets/style.css
docs/MODEL_CARD.md
model/mp_rf_model.joblib
model/metadata.json
```
