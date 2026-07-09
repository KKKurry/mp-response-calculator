# 新手部署说明

## 使用 GitHub Desktop 更新线上网站

1. 解压 `mp_response_calculator_v15_binary_threshold.zip`
2. 打开 GitHub Desktop 管理的本地仓库文件夹
3. 复制以下内容到本地仓库并覆盖同名文件：

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

如果原仓库已有 `model/mp_rf_model.joblib`，可以不复制 `model` 文件夹。

4. 回到 GitHub Desktop，填写 Summary：

```text
Update v15 binary threshold UI
```

5. 点击 `Commit to main`
6. 点击 `Push origin`
7. 回到 Streamlit，点击 `Manage app → Reboot app`
8. 等 1–3 分钟后使用 `Ctrl + F5` 强制刷新网页。

## 重要说明

V15 前端显示的是二分类判定，不是临床真实概率。模型输出值仅用于与固定阈值比较。
