# FinalProject  House Price Prediction: Advanced Regression Techniques

## 1. Problem and Data Overview  
**Challenge**: This project goal is to predict the final sale price of residential homes using features such as physical attributes, location, construction year, and remodeling details. This is a supervised regression task requiring handling of missing values, feature engineering, and model optimization.

**Dataset**:  
- **Size**:  
  - Training set: 1,460 samples with 80 features.  
  - Test set: 1,459 samples (unlabeled).  
- **Features**:  
  - Numerical (e.g., `GrLivArea`, `TotalBsmtSF`).  
  - Categorical (e.g., `Neighborhood`, `MSZoning`).  
- **Target Variable**: `SalePrice` (house price).  

---

## 2. Exploratory Data Analysis (EDA)  
### **Data Investigation and Visualization**  
1. **Target Distribution**:  
   - `SalePrice` showed a right-skewed distribution. After log transformation (`log1p`), it approximated a normal distribution.  
2. **Missing Values**:  
   - High missing rates in `PoolQC` (99%), `MiscFeature` (96%).  
   - Filled numerical features with median values and categorical features with modes.  
3. **Feature Correlation**:  
   - Top 10 features correlated with `SalePrice`: `OverallQual`, `GrLivArea`, `GarageCars`, `GarageArea`, `TotalBsmtSF`, `1stFlrSF`, `FullBath`, `TotRmsAbvGrd`, `YearBuilt`, `YearRemodAdd`.  

### **Data Cleaning and Plan**  
- **Handling Missing Values**: Median/mode imputation.  
- **Feature Selection**: Used top 10 correlated features to reduce dimensionality.  
- **Future Steps**: Explore feature interactions, nonlinear models (e.g., gradient boosting).  

---

## 3. Model Architectures and Comparison  
### **Model Selection**  
| Model                          | Key Characteristics                                                                 | Rationale                                      |  
|--------------------------------|-------------------------------------------------------------------------------------|------------------------------------------------|  
| **Linear Regression (LR)**    | Simple, interpretable. Assumes linear relationships.                               | Baseline for quick validation.                |  
| **Random Forest (RF)**         | Ensemble of trees, captures nonlinearity, robust to overfitting.                   | Suitable for high-dimensional data.            |  
| **Optimized Random Forest**    | Hyperparameter tuning (e.g., `n_estimators=200`, `max_depth=20`) via grid search.   | Balances model complexity and generalization.  |  

### **Loss Function and Hyperparameter Tuning**  
- **Loss Function**: Mean Squared Error (MSE), chosen for sensitivity to outliers and direct alignment with squared errors in price prediction.  
- **Hyperparameter Tuning**: Grid search optimized `n_estimators` and `max_depth`, reducing overfitting.  

### **Training Results Visualization**  
- Random Forest predictions align closer to the true values (diagonal line).  

---

## 4. Results and Analysis  
### **Performance Comparison**  
| Model                     | MSE             | RMSE     | R²     | Cross-Validation RMSE (Std. Dev)         |  
|---------------------------|-----------------|----------|--------|------------------------------------------|  
| Linear Regression         | 1.56×10⁹        | 39,474   | 0.80   | 38,573 (±6,480)                          |  
| Random Forest             | 8.77×10⁸        | 29,619   | 0.89   | 32,154 (±4,666)                          |  
| Optimized Random Forest   | **8.73×10⁸**    | **29,546** | **0.89** | **31,947 (±4,533)**                      |  

### **Key Insights**  
1. **Model Superiority**:  
   - Random Forest outperformed LR due to nonlinear relationships (e.g., `YearBuilt` vs `SalePrice`).  
   - Tuned RF achieved lower cross-validation variance, indicating improved stability.  

2. **Hyperparameter Impact**:  
   - Higher `n_estimators` and constrained `max_depth` reduced overfitting.  

---

## 5. Conclusion
### **Summary**  
- **Best Model**: Optimized Random Forest (RMSE=29,546), 25% improvement over LR.  
- **Feature Selection**: Top 10 features maintained efficiency without significant accuracy loss.  

### **Future Improvements**  
1. **Feature Engineering**:  
   - Add interaction terms (e.g., `GrLivArea × OverallQual`).  
   - Apply advanced encoding (e.g., target encoding) for categorical features.  

2. **Model Enhancement**:  
   - Test gradient boosting (XGBoost, LightGBM) or stacking ensembles.  
