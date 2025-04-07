# Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset
df = pd.read_csv(r"/Users/gio/Downloads/EU_Startup_DealSourcing.csv")

# Show head
print(df.head())

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