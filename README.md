# flood-prediction-ml-competition
This ML project focuses on flood risk prediction using structured data containing environmental, climatic, and infrastructural indicators. The data was originally developed for a Kaggle competition, but I began this project after the competition had ended.

I engineered a set of statistical features to capture distributional characteristics, including the mean, median, standard deviation, min, max, quartiles, and total sum across all features. These were intended to model latent patterns that span across multiple input variables. After comparative model evaluation, I selected CatBoost for its superior performance relative to XGBoost.

Model performance was assessed using 5-fold cross-validation with R² as the evaluation metric. The final model was retrained on the full training set before generating test predictions for submission. The final model had an $R^2$ score of 0.86901.
