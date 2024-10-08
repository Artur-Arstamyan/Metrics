Data normalization is a common preprocessing step in machine learning (ML) where numerical features are scaled to ensure they have a similar range. This is crucial for algorithms that are sensitive to the scale of input features.
Here are **machine learning algorithms that typically require data normalization:**

**1. K-Nearest Neighbors (KNN):**
- **Why**: KNN uses distance-based calculations (e.g., Euclidean distance). If one feature has a larger range, it will dominate the distance calculations.
- **Normalization**: Ensures that all features contribute equally to distance measurements.  
**2. Support Vector Machines (SVM):**
- **Why**: SVM attempts to find a hyperplane that maximizes the margin between classes. If features are on different scales, the hyperplane will be distorted.
- **Normalization**: Helps the SVM to weigh features equally when defining the decision boundary.  
**3. Logistic Regression:**
- **Why**: Although Logistic Regression doesn't explicitly depend on distances, large feature ranges can disproportionately influence the model’s coefficients.
- **Normalization**: Ensures balanced contributions of all features during optimization.  
**4. K-Means Clustering:**
- **Why**: K-Means uses Euclidean distance to assign data points to clusters. Features with larger ranges can dominate the distance computation.
- **Normalization**: Prevents any single feature from overly influencing the cluster assignments.  
**5. Principal Component Analysis (PCA):**
- **Why**: PCA identifies directions (principal components) of maximum variance. If features are on different scales, the components will be biased towards higher-variance features.
- **Normalization**: Ensures each feature contributes proportionally to the principal components.  
**6. Neural Networks (including Deep Learning):**
- **Why**: Neural networks involve weight optimization, and if features are on different scales, gradients may become too large or small, slowing down convergence or leading to poor optimization.
- **Normalization**: Helps the network learn faster and more effectively by keeping gradients in similar ranges.  
**7. Gradient Descent-based Algorithms (e.g., Linear Regression, Logistic Regression):**
- **Why**: Algorithms that rely on gradient descent optimize over feature space. Large feature scales can lead to uneven updates and slow convergence.
- **Normalization**: Ensures that gradient updates are balanced and that optimization proceeds efficiently.  
**8. Ridge and Lasso Regression (Regularized Linear Models):**
- **Why**: These algorithms introduce penalty terms (L1 or L2 regularization), and different feature scales can result in biased penalty terms.
- **Normalization**: Balances the regularization penalty across all features.  
**9. Perceptron:**
- **Why**: Like other linear classifiers, the Perceptron algorithm is affected by different feature scales when calculating the decision boundary.
- **Normalization**: Ensures equal contribution of all features when updating weights.  
**Algorithms That Don’t Typically Require Normalization:**
Not all algorithms are sensitive to feature scaling. For example:   

- **Tree-based algorithms** (e.g., Decision Trees, Random Forests, Gradient Boosting): These algorithms are not distance-based and don’t rely on feature scaling.
- **Naive Bayes**: It works with probabilities and isn’t affected by feature scaling.  
**Common Normalization Techniques:**
- **Min-Max Scaling**: Scales features to a fixed range, typically [0, 1].
- **Z-score Standardization**: Centers data to have a mean of 0 and a standard deviation of 1.
- **Robust Scaling**: Uses the median and interquartile range to scale data, making it robust to outliers.   

**Summary:**
**Normalization is crucial** for algorithms like KNN, SVM, K-Means, and Neural Networks that depend on distance, gradient-based optimization, or variance.
Tree-based models and Naive Bayes, however, do not require normalization.
