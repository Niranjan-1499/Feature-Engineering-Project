# Feature-Engineering-Project

Feature Scaling
-
- Feature scaling is the process of transforming numerical features in your dataset so they are on a similar scale.
- It’s important because:
  - Many algorithms (like KNN, SVM, logistic regression, neural networks, gradient descent-based models) are sensitive to the magnitude of features.
  - Without scaling, features with large numerical ranges can dominate distance-based metrics or optimization steps.

  <img width="741" height="355" alt="image" src="https://github.com/user-attachments/assets/9414e251-a9ab-4ef5-b7dd-98cf0976881c" />
  
**When to scale**
- 
- Distance-based algorithms (KNN, K-means, SVM, PCA)
- Gradient descent-based models (linear regression, logistic regression, neural networks)
- Not always necessary for tree-based models (Decision Tree, Random Forest, XGBoost), because they split based on thresholds not distances.

Understanding of the data
-
  <img width="1152" height="717" alt="image" src="https://github.com/user-attachments/assets/44ee852a-819a-42e6-bdd6-55452660984e" />

- n this step, we begin by importing essential Python libraries such as NumPy and Pandas for data manipulation, Matplotlib and Seaborn for visualization, and    various Scikit-learn modules for preprocessing, model building, and evaluation.
- The dataset Social_Network_Ads_1.csv is then loaded into a Pandas DataFrame, providing information about users’ demographics and purchasing behavior. A random sample of three rows is displayed to quickly inspect the data structure. Since the User ID and Gender columns are not relevant for the prediction task, they are removed using the .iloc method, retaining only the numerical features Age and EstimatedSalary, along with the target variable Purchased.
- This refined dataset ensures that only meaningful predictors are passed into the machine learning pipeline, preparing it for the next stage, which involves feature scaling and model training. 


