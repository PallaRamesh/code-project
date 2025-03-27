import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# 1. Load the dataset directly from the UCI Machine Learning repository URL
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"
data = pd.read_csv(url, sep=";", header=0)

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# 2. Data Preprocessing

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Convert categorical variables to numerical using one-hot encoding
data = pd.get_dummies(data, drop_first=True)

# Separate features (X) and target variable (y)
X = data.drop('y_yes', axis=1)  # 'y_yes' is the target variable (subscribed to term deposit)
y = data['y_yes']  # Target: 1 if subscribed, 0 otherwise

# 3. Feature Scaling (Optional but recommended for some models)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. Train the Decision Tree Classifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 6. Evaluate the Model

# Predict the target variable for the test set
y_pred = model.predict(X_test)

# Accuracy of the model
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")

# Classification report (Precision, Recall, F1-score)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 7. Visualize the Decision Tree (Optional)
plt.figure(figsize=(15,10))
plot_tree(model, filled=True, feature_names=X.columns, class_names=["No", "Yes"], rounded=True)
plt.title("Decision Tree Visualization")
plt.show()
