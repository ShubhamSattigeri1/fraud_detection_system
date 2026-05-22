import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import shap
df = pd.read_csv('creditcard.csv')
features = [f'V{i}' for i in range(1, 29)] 
X = df[features]
y = df['Class']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    class_weight='balanced',
    random_state=42
)
rf_model.fit(X_train, y_train)
rf_accuracy = rf_model.score(X_test, y_test)
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, rf_model)
print(f"Model Accuracy Random : {rf_accuracy:.2%}")
