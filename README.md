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

- In this step, we begin by importing essential Python libraries such as NumPy and Pandas for data manipulation, Matplotlib and Seaborn for visualization, and    various Scikit-learn modules for preprocessing, model building, and evaluation.
- The dataset Social_Network_Ads_1.csv is then loaded into a Pandas DataFrame, providing information about users’ demographics and purchasing behavior. A random sample of three rows is displayed to quickly inspect the data structure. Since the User ID and Gender columns are not relevant for the prediction task, they are removed using the .iloc method, retaining only the numerical features Age and EstimatedSalary, along with the target variable Purchased.
- This refined dataset ensures that only meaningful predictors are passed into the machine learning pipeline, preparing it for the next stage, which involves feature scaling and model training.

Train test split
-

<img width="1167" height="767" alt="image" src="https://github.com/user-attachments/assets/dd9dd194-b322-49ab-b272-209b4d8118c3" />

<img width="1172" height="686" alt="image" src="https://github.com/user-attachments/assets/ef62adef-f6c3-4d84-8a19-8c54f186112f" />

- In this step, the dataset is divided into training and testing sets using the train_test_split function from Scikit-learn. The target variable Purchased is separated from the predictor variables Age and EstimatedSalary, with 80% of the data allocated for training and 20% for testing to evaluate model performance on unseen data. 
- Feature scaling is then applied using StandardScaler, which standardizes each feature by removing the mean and scaling to unit variance. The scaler is fitted only on the training data to prevent data leakage, and the same transformation is applied to the test set.
- Before scaling, the features had their original ranges (e.g., salaries ranging from 15,000 to 150,000), but after scaling, they are centered around a mean of 0 with a standard deviation of 1.
- This transformation ensures that both features contribute equally to the model, particularly benefiting algorithms sensitive to feature magnitude. The descriptive statistics confirm that scaling has adjusted the values appropriately, with standardized minimum, maximum, and quartile ranges.

Effect of Scaling
-

<img width="1168" height="561" alt="image" src="https://github.com/user-attachments/assets/a5b7926f-02ff-4d5d-98b3-eea975cbc7b3" />

-This visualization compares the feature values before and after applying StandardScaler. On the left, the scatter plot shows the original data distribution of Age and EstimatedSalary in their raw form, where the salary values span a much larger range than age, leading to a scale imbalance. 
- On the right, the same data is plotted after scaling, where both features have been standardized to have a mean of 0 and a standard deviation of 1. This transformation changes only the scale and center of the data, not the relative positioning of the points, ensuring that both features contribute equally during model training.
- Such scaling is particularly important for algorithms that are sensitive to feature magnitude, such as logistic regression, SVM, and KNN.

<img width="1160" height="647" alt="image" src="https://github.com/user-attachments/assets/9018fbf0-89eb-4e70-9e54-cf962f65c273" />

- This figure shows the kernel density estimation (KDE) plots of the features Age and EstimatedSalary before and after applying StandardScaler.
- In the left plot, before scaling, the EstimatedSalary feature dominates the horizontal axis due to its much larger numeric range compared to Age, making the Age distribution appear compressed near zero. In the right plot, after scaling, both features are transformed to have a mean of 0 and a standard deviation of 1. This results in both distributions being centered around zero and comparable in scale, while preserving their shapes.
- This standardization ensures that both features contribute equally during model training, which is especially important for algorithms that rely on distance calculations or assume features are on the same scale.




