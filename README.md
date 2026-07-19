# Insurance Medical Expenses — EDA & Regression Mini Project

Predicting medical insurance expenses using Linear Regression and Polynomial Regression, with a full exploratory data analysis (EDA) and preprocessing pipeline.

## Dataset

- **Source:** [Insurance Premium Prediction](https://www.kaggle.com/datasets/noordeen/insurance-premium-prediction) (Kaggle)
- **Rows:** 1,338 | **Columns:** 7
- **Features:** `age`, `sex`, `bmi`, `children`, `smoker`, `region`
- **Target:** `expenses` (medical insurance cost)
- No missing values; all columns had correct datatypes.

## Algorithms Used

| Algorithm | Description | When to use |
|---|---|---|
| **Linear Regression** | Finds the best-fit line minimizing the difference between predicted and actual values | Target is continuous and the relationship with features is linear |
| **Polynomial Regression** | Extends Linear Regression by adding polynomial terms (degree 2) to model non-linear relationships | Relationship between features and target is non-linear |

**Advantages:** simple, fast to train, easy to interpret.
**Limitations:** assumes a linear relationship (Linear Regression), sensitive to outliers, may underperform on complex datasets.

## Exploratory Data Analysis

- **Expenses:** right-skewed — most people have low medical expenses, a small group has very high expenses (confirmed via histogram and box plot, which also revealed several outliers).
- **Age:** ranges 18–64, fairly evenly represented.
- **BMI:** most values fall between 25–35.
- **Children:** most people have none.
- **Age vs. Expenses:** weak positive relationship — expenses tend to rise with age.
- **BMI vs. Expenses:** weak positive relationship, more noticeable above BMI 30.
- **BMI Category vs. Expenses:** expenses increase with BMI category; the Obese group has the highest expenses and most outliers.
- **Smoker vs. Expenses:** smokers have noticeably higher expenses than non-smokers, with several high-end outliers.
- **Correlation Heatmap:** smoking shows the strongest positive correlation with expenses; age and BMI show moderate positive correlations.

## Data Preprocessing

1. **Missing values:** none found — confirmed via `isnull().sum()` and `info()`.
2. **Encoding categorical variables:**
   - `sex`, `smoker` → Label Encoding (binary categories, 0/1)
   - `region` → One-Hot Encoding (4 unordered categories, no natural rank)
3. **Train/test split:** 80/20, `random_state=42`, done before scaling to avoid data leakage.
4. **Feature scaling:** `StandardScaler` fit on `X_train` only, then applied to `X_test` via `.transform()`.
5. **Target scaling:** `expenses` also standardized with a separate `StandardScaler`; predictions are inverse-transformed back to dollars before evaluation so error metrics stay interpretable.

## Modeling

- **Linear Regression** — trained on the scaled features.
- **Polynomial Regression (degree 2)** — `PolynomialFeatures(degree=2)` + `LinearRegression` via `make_pipeline`.

## Results

| Metric | Linear Regression | Polynomial Regression |
|---|---|---|
| MAE | $4,181 | $2,729 |
| RMSE | $5,796 | $4,550 |
| R² Score | 0.78 | 0.87 |

Polynomial Regression outperformed Linear Regression on every metric, confirming the relationship between features (especially age and BMI) and expenses is non-linear rather than a straight line.

## Key Insights

- **Smoking is the dominant cost driver** — smoking status outweighs all other factors in predicting medical expenses.
- **Age and BMI matter, but modestly** — both show weak-to-moderate positive correlation, with BMI's effect more visible past 30.
- **A small high-cost minority skews the data** — concentrated among smokers and the obese, driving the right-skew and most outliers.
- **The non-linear model fits better** — adding a degree-2 polynomial term meaningfully improved R² (0.78 → 0.87), confirming a curved rather than linear relationship.
- **Errors dropped substantially with Polynomial Regression** — MAE fell from $4,181 to $2,729, and RMSE from $5,796 to $4,550.


## **Team**
- Raghad Alharbi
- Maram Alzahrani
- Turki Abuhaimed
- Mohammed Alshatri

## Tech Stack

`pandas` · `numpy` · `matplotlib` · `seaborn` · `scikit-learn` (`LinearRegression`, `PolynomialFeatures`, `LabelEncoder`, `StandardScaler`, `train_test_split`, `Ridge`, `Lasso`, `cross_val_score`)
