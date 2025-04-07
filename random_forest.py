# Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset
df = pd.read_csv("EU_Startup_DealSourcing.csv")

# Show head
print(df.head())

# Funding Raised distribution

plt.figure(figsize=(8, 5))
sns.histplot(df['funding_raised'], bins=30, kde=True)
plt.title("Distribution of Funding Raised")
plt.xlabel("Revenue")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# correlation matrix

plt.figure(figsize=(12, 10))
numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()

sns.heatmap(corr_matrix, cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap (Numerical Features)")
plt.tight_layout()
plt.show()

# Outliers identification

plt.figure(figsize=(8, 5))
sns.boxplot(x=df['funding_raised'])
plt.title("Boxplot of Funding Raised")
plt.xlabel("Funding")
plt.tight_layout()
plt.show()

# Outlier ratio for numerical variables
def outlier_percentage(df):
    outlier_stats = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        percentage = 100 * len(outliers) / len(df)
        outlier_stats[col] = round(percentage, 2)
    
    return pd.Series(outlier_stats, name="Outlier %")

# Show results
outlier_percentages = outlier_percentage(df)
print(outlier_percentages.sort_values(ascending=False))

# Preprocessing
df = df.drop(columns=['company_name'])

# Separate target variable
y = df['high_potential']
X = df.drop(columns=['high_potential'])

# One-hot encoding on categorical variables
X = pd.get_dummies(X, columns=['country', 'sector'], drop_first=True)

# Split train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Feature importance visualisation
importances = model.feature_importances_
feat_names = X.columns
feat_imp = pd.Series(importances, index=feat_names).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=feat_imp[:10], y=feat_imp.index[:10])
plt.title("Top 10 Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

import joblib

# Save file
save_path = "/Users/gio/Desktop/TRINITY COLLEGE/MODULES/BIG DATA & AI/INDIVIDUAL PROJECT/new project"

# Save model
joblib.dump(model, f"{save_path}/deal_sourcing_model.pkl")

# Save features columns
joblib.dump(X.columns.tolist(), f"{save_path}/feature_columns.pkl")
