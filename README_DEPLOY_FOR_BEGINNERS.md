# 新手部署指南：MP Response Calculator

这个项目是 Streamlit 网站。部署到公网推荐使用 Streamlit Community Cloud。

## 你需要准备

1. 一个 GitHub 账号
2. 一个 Streamlit Community Cloud 账号（可用 GitHub 登录）
3. 本文件夹里的所有项目文件

## 一、先在本地确认能运行

在本文件夹中打开命令行：

```bash
pip install -r requirements.txt
streamlit run app.py
```

浏览器打开：

```text
http://localhost:8501
```

## 二、上传到 GitHub

1. 打开 https://github.com
2. 点击右上角 `+`
3. 选择 `New repository`
4. Repository name 填：`mp-response-calculator`
5. 建议先选 `Private`，如果你希望公开展示可以选 `Public`
6. 不要勾选 Add a README file
7. 点击 `Create repository`
8. 进入新仓库后，点击 `uploading an existing file` 或 `Add file` → `Upload files`
9. 打开本文件夹，选中所有文件和文件夹，拖入 GitHub 上传区域
10. 点击底部绿色按钮 `Commit changes`

## 三、部署到 Streamlit Cloud

1. 打开 https://share.streamlit.io
2. 用 GitHub 账号登录
3. 点击右上角 `Create app`
4. 选择 `Deploy a public app from GitHub` 或连接你的 GitHub 仓库
5. Repository 选择：`你的用户名/mp-response-calculator`
6. Branch 选择：`main`
7. Main file path 填：`app.py`
8. 在 Advanced settings 中，Python version 建议选择 `3.11` 或 `3.12`
9. 点击 `Deploy`

## 四、部署成功后

你会得到一个公网网址，类似：

```text
https://mp-response-calculator.streamlit.app
```

把这个链接复制出来，就是你的 Website。

## 常见错误

### 1. ModuleNotFoundError

通常是 `requirements.txt` 没有上传，或不在项目根目录。

### 2. FileNotFoundError: model/mp_rf_model.joblib

说明 `model` 文件夹没有上传完整。请确认 GitHub 上存在：

```text
model/mp_rf_model.joblib
```

### 3. 进入网站后页面报错

先点击 Streamlit Cloud 右下角或管理页面的日志，复制红色错误信息给我。

