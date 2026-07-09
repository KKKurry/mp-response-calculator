# Model Card: MRI-IHC MP Response Calculator

## Intended use

This model is designed as a research-use web calculator to estimate the probability of favourable pathological response, defined as Miller–Payne grade 4–5, before treatment.

## Input variables

The model uses pretreatment variables only:

- Age
- Calcification status
- DCE-MRI curve pattern
- ADC value
- Maximum tumour diameter in cm
- ER percentage
- PR percentage
- HER2 score
- Ki-67 percentage

Excluded fields include patient names, examination dates, original MP grade, and molecular subtype.

## Model family

Calibrated Random Forest with median imputation and rule-based clinical feature engineering.

## Output

The website returns the estimated MP 4–5 response probability, a likelihood tier, and a predicted MP response group for research presentation.

## Limitations

- The current package is based on the supplied dataset and saved model artifact.
- Independent temporal/external validation is recommended before clinical interpretation.
- The app should not be used as a standalone clinical decision-making system.
