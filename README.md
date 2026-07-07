# MRI–IHC MP Response Calculator · v9 Balanced Enriched

A Streamlit web calculator for estimating favorable Miller–Payne 4–5 response using pre-treatment clinical, MRI and IHC features.

## v9 design changes

- Balanced laboratory web-server style inspired by bioinformatics tool portals.
- Less cramped than v8, but with reduced empty space compared with v7.
- Enriched content blocks: clinical question, feature scope, result format and research-use boundary.
- Improved feature-input form with section labels, explanatory notes and input help.
- Right-side interpretation panel added to make the page more informative.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

Upload the visible files and folders to the root of your GitHub repository, then reboot the Streamlit app.

Required root structure:

```text
app.py
model_pipeline.py
requirements.txt
assets/
docs/
model/
sample_input.csv
sample_input.xlsx
README.md
```
