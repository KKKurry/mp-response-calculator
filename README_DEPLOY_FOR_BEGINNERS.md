# 新手部署说明

1. 解压本压缩包。
2. 将文件复制到本地 GitHub 仓库文件夹。
3. 在 GitHub Desktop 中填写 Summary，例如：`Update v12 warm medical dashboard UI`。
4. 点击 `Commit to main`。
5. 点击 `Push origin`。
6. 回到 Streamlit Cloud，点击 `Manage app → Reboot app`。
7. 等待 1–3 分钟后用 `Ctrl + F5` 强制刷新。

若网页上传频繁崩溃，建议只上传：`app.py`、`model_pipeline.py`、`assets`、`docs`，不要重复上传 `model` 文件夹。
