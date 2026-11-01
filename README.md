# House Rent Prediction

A regression project that predicts the rent of houses, apartments, and flats across major Indian cities. The model is a **CatBoost regressor** wrapped in a scikit-learn pipeline, with the target (rent) log-transformed to handle its heavily skewed distribution.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Approach](#approach)
- [Results](#results)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Requirements](#requirements)
- [Possible Improvements](#possible-improvements)
- [Acknowledgements](#acknowledgements)

## Overview

| | |
|---|---|
| **Task** | Regression: predict the `Rent` of a rental listing |
| **Model** | `CatBoostRegressor` inside a scikit-learn `Pipeline` |
| **Target transform** | `log1p` for training, `expm1` to convert predictions back to rupees |
| **Evaluation** | Predicted vs. actual rent on a held-out test set (949 listings) |

Rent prices vary enormously between cities and property types, and most listings sit at the low end with a long tail of expensive properties. Training on the log of the rent makes the target more evenly distributed, which usually helps tree-based models learn the pattern more reliably.

## Dataset

**Source:** [House Rent Prediction Dataset on Kaggle](https://www.kaggle.com/datasets/iamsouravbanerjee/house-rent-prediction-dataset) by Sourav Banerjee

The dataset contains about 4,700 rental listings (houses, apartments, and flats) across six Indian cities: Bangalore, Chennai, Delhi, Hyderabad, Kolkata, and Mumbai. It has 12 columns:

| Column | Description |
|---|---|
| `Posted On` | Date the listing was posted |
| `BHK` | Number of bedrooms, hall, and kitchen |
| `Rent` | Rent of the property in Indian rupees (**target**) |
| `Size` | Size of the property in square feet |
| `Floor` | Floor the property is on and total floors in the building (e.g. `Ground out of 2`, `1 out of 3`) |
| `Area Type` | How the size is measured (`Super Area`, `Carpet Area`, or `Built Area`) |
| `Area Locality` | Locality of the property |
| `City` | City where the property is located |
| `Furnishing Status` | `Furnished`, `Semi-Furnished`, or `Unfurnished` |
| `Tenant Preferred` | Type of tenant preferred (`Bachelors`, `Family`, or `Bachelors/Family`) |
| `Bathroom` | Number of bathrooms |
| `Point of Contact` | Whom to contact about the listing (e.g. `Contact Owner`, `Contact Agent`) |

To run the notebook, download `House_Rent_Dataset.csv` from the Kaggle link above and place it in the `data/` folder. Please check the dataset's page on Kaggle for its license and attribution terms.

## Approach

1. **Load the data** with pandas and split it into training and testing sets (949 listings are held out for testing).
2. **Preprocess the features** with a scikit-learn preprocessing step (`preprocessed`) that is included as the first stage of the pipeline, so exactly the same transformations are applied to the training and test data.
3. **Train a CatBoost regressor** (`random_state=43`) as the final stage of the pipeline.
4. **Transform the target.** The whole pipeline is wrapped in a `TransformedTargetRegressor`, which fits the model on `log1p(rent)` and automatically converts predictions back to rupees with `expm1`.

```python
cat = TransformedTargetRegressor(
    regressor=Pipeline([
        ('pipe', preprocessed),
        ('cats', CatBoostRegressor(random_state=43, verbose=0))
    ]),
    func=np.log1p,
    inverse_func=np.expm1
)

cat.fit(x_train, y_train)
y_pred = cat.predict(x_test)
```

5. **Evaluate** by comparing predicted and actual rent in a results table and a scatter plot.

## Results

### Sample predictions

A comparison of actual and predicted rent (in rupees) on the first few test listings:

| Actual | Predicted | Error (predicted − actual) |
|---:|---:|---:|
| 4,000 | 10,562 | 6,562 |
| 7,500 | 7,597 | 97 |
| 13,000 | 11,604 | −1,396 |
| 20,000 | 13,472 | −6,528 |
| 7,000 | 7,064 | 64 |

### Evaluation metrics

| Metric | Value |
|---|---|
| MAE | _add value_ |
| RMSE | _add value_ |
| R² | _add value_ |

### Predicted vs. actual rent

![Predicted vs Actual Rent](images/predicted_vs_actual.png)

What the plot shows:

- For typical rents (roughly under 50,000), predictions sit close to the red "Perfect Prediction" line, which is where most of the listings are.
- As actual rent increases, the points spread out more and the model tends to **under-predict** expensive properties. High-end listings are rare in the data, so the model has fewer examples to learn from.
- The axes are limited to the 99th percentile of actual rent so that a few extreme outliers don't squash the rest of the plot.

## Project Structure

```
house-rent-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── your-notebook-name.ipynb      # main notebook
├── images/
│   └── predicted_vs_actual.png   # plot used in this README
└── data/
    └── House_Rent_Dataset.csv    # download from Kaggle
```

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/YOUR-USERNAME/YOUR-REPO.git
   cd YOUR-REPO
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/iamsouravbanerjee/house-rent-prediction-dataset) and place `House_Rent_Dataset.csv` in the `data/` folder.
4. Launch Jupyter and run all cells:

   ```bash
   jupyter notebook
   ```

## Requirements

- Python 3.9+
- pandas
- numpy
- scikit-learn
- catboost
- matplotlib
- jupyter

Install everything with:

```bash
pip install pandas numpy scikit-learn catboost matplotlib jupyter
```

## Possible Improvements

- Report standard regression metrics (MAE, RMSE, R²) alongside the visual comparison, and add cross-validation for a more reliable estimate.
- Tune CatBoost hyperparameters (depth, learning rate, iterations, regularization).
- Pass categorical columns such as `City`, `Area Locality`, and `Furnishing Status` directly to CatBoost through `cat_features` instead of encoding them manually.
- Engineer extra features, such as the floor number and building height extracted from the `Floor` column, or rent per square foot by city.
- Investigate high-rent outliers, which are currently under-predicted, for example by using a loss function that is more robust to outliers or by modelling each city separately.
- Compare against other models such as LightGBM, XGBoost, and Random Forest.
- Inspect feature importance to see what drives rent the most.

## Acknowledgements

- Dataset: [House Rent Prediction Dataset](https://www.kaggle.com/datasets/iamsouravbanerjee/house-rent-prediction-dataset) by Sourav Banerjee on Kaggle.
- Model built with [CatBoost](https://catboost.ai/) and [scikit-learn](https://scikit-learn.org/).
