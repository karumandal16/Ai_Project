import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

# Load dataset
data = pd.read_csv("data.csv")
X = data[['age','tenure','usage']]
y = data['churn']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# Log model & metrics in MLflow
mlflow.sklearn.log_model(model, "customer_churn_model")
mlflow.log_metric("accuracy", accuracy)
