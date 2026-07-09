# 部署说明

将本文件夹内容复制到 GitHub 仓库根目录，提交到 `main` 分支后，在 Streamlit Cloud 中重启应用。

建议通过 GitHub Desktop 更新：

1. 复制 `app.py`、`model_pipeline.py`、`assets`、`docs`、`README.md`、`requirements.txt` 等文件到本地仓库。
2. 如原仓库已有 `model/mp_rf_model.joblib`，可不重复复制 `model` 文件夹。
3. Commit to main。
4. Push origin。
5. Streamlit Cloud 中 Reboot app。
