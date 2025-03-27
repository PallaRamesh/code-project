import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Data Loading and Initial Overview
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
data = pd.read_csv(url)

# Show the first few rows of the dataset
print(data.head())

# Get a summary of the dataset
print("\nData Info:")
print(data.info())

# Get basic statistics
print("\nDescriptive Statistics:")
print(data.describe())

# 2. Data Cleaning

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Handle missing values
# Fill missing 'Age' values with the median
data['Age'].fillna(data['Age'].median(), inplace=True)

# Drop rows where 'Embarked' is missing (optional, based on your analysis)
data.dropna(subset=['Embarked'], inplace=True)

# Check for duplicates
print(f"\nDuplicate Rows: {data.duplicated().sum()}")

# Drop duplicates if any
data.drop_duplicates(inplace=True)

# Convert 'Sex' column to numerical (Female=0, Male=1)
data['Sex'] = data['Sex'].map({'female': 0, 'male': 1})

# Drop unnecessary columns like 'Name' and 'Ticket'
data.drop(['Name', 'Ticket'], axis=1, inplace=True)

# Check if there are still any missing values
print("\nMissing Values After Cleaning:")
print(data.isnull().sum())

# 3. Exploratory Data Analysis (EDA)

# Set up the plotting style
sns.set(style="whitegrid")

# 3.1 Visualize the distribution of Age
plt.figure(figsize=(10, 6))
sns.histplot(data['Age'], kde=True, bins=30)
plt.title('Age Distribution')
plt.show()

# 3.2 Visualize Survival Rate
plt.figure(figsize=(10, 6))
sns.countplot(x='Survived', data=data, palette='Set2')
plt.title('Survival Count')
plt.show()

# 3.3 Survival rate by Pclass
plt.figure(figsize=(10, 6))
sns.countplot(x='Pclass', hue='Survived', data=data, palette='Set2')
plt.title('Survival Count by Pclass')
plt.show()

# 3.4 Survival rate by Sex
plt.figure(figsize=(10, 6))
sns.countplot(x='Sex', hue='Survived', data=data, palette='Set2')
plt.title('Survival Count by Sex')
plt.show()

# 3.5 Survival rate by Age
plt.figure(figsize=(10, 6))
sns.boxplot(x='Survived', y='Age', data=data, palette='Set2')
plt.title('Survival by Age')
plt.show()

# 3.6 Survival rate by Embarked
plt.figure(figsize=(10, 6))
sns.countplot(x='Embarked', hue='Survived', data=data, palette='Set2')
plt.title('Survival Count by Embarked')
plt.show()

# 3.7 Correlation Heatmap
correlation = data.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()

# 4. Conclusion
print("\nConclusion of EDA:")
print("1. Passengers in Pclass 1 had the highest survival rate, followed by Pclass 2 and Pclass 3.")
print("2. Females had a significantly higher survival rate than males.")
print("3. Younger passengers had a higher chance of survival.")
print("4. Passengers who boarded from Cherbourg (C) had a higher survival rate.")
print("5. Correlation analysis shows that 'Fare' and 'Pclass' have a higher correlation with 'Survived'.")
