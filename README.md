# House-Price-Prediction-using-Machine-Learning-with-Python

=> Dataset
- The dataset contains the following features:

  - MedInc: Median income in block group.
  - HouseAge: Median house age in block group.
  - AveRooms: Average number of rooms per household.
  - AveBedrms: Average number of bedrooms per household.
  - Population: Block group population.
  - AveOccup: Average number of household members.
  - Latitude: Block group latitude.
  - Longitude: Block group longitude.
- The target variable is Price, which represents the median house value in $100,000 units.

=> Model Training and Evaluation
- The XGBoost Regressor is used to train the model. The dataset is split into training and testing sets to evaluate the performance of the model. Metrics used for evaluation:

-  R-Squared Error: Measures the percentage of the variance in the target that the model explains.
- Mean Absolute Error (MAE): Measures the average magnitude of errors in predictions.

=> Training Performance:
- R-Squared: 0.94
- MAE: 0.19
 
=> Testing Performance:
- R-Squared: 0.83
- MAE: 0.31

=> Visualizations
- Correlation Heatmap: Visualizes the correlation between the different features in the dataset.
- Scatter Plots: Compare Actual Prices vs Predicted Prices for both training and testing data.

=> Results
- The model performs well on the dataset, achieving an R-squared score of 0.83 on the testing set. The scatter plots demonstrate a strong alignment between actual and predicted prices.

=> Technologies Used
- Python: Programming language.
- XGBoost: Regressor model for predicting house prices.
- NumPy: Numerical operations.
- Pandas: Data manipulation and analysis.
- Matplotlib & Seaborn: Data visualization libraries.
- Scikit-learn: Tools for model evaluation and dataset handling.

=> Future Enhancements
- Hyperparameter Tuning: Optimize XGBoost hyperparameters to improve model performance.
- Feature Engineering: Introduce new features to enhance predictive accuracy.
- Comparison with Other Models: Implement and compare performance with other regression models like Random Forest, Linear Regression, etc.
