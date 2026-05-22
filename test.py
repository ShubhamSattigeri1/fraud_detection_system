import pandas as pd
import numpy as np
import shap
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

df = pd.read_csv('creditcard.csv')
features = [f'V{i}' for i in range(1, 29)] 
X = df[features]
y = df['Class']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_dict = {
    'max_depth' : randint(3,100),
    'random_state' : randint(1, 44)
}
random_search = RandomizedSearchCV(estimator = RandomForestClassifier(), param_distributions=param_dict, cv=5, random_state=42)
random_search.fit(X_train, y_train)
print(random_search.best_params_)
clf_pipeline = RandomForestClassifier()

smote = SMOTE(sampling_strategy="auto", random_state=42, k_neighbors=5)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

numerical_col = ["V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28"]
numerical_transformer = Pipeline(
    steps = [('input', SimpleImputer
              (strategy="median")),
              ('scalar', StandardScaler())]) 
preprocessor = ColumnTransformer(
    transformers = [('num', StandardScaler(), numerical_col)] 
)
clf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model'
                               'model', RandomForestClassifier(
                                   random_state=42))] )
clf_pipeline.fit(X_train_sm, y_train_sm)
y_pred = clf_pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

explainer = shap.TreeExplainer(clf_pipeline)
shap_values = explainer(X_test)

shap.plots.waterfall(shap_values[0, :, 1])
shap.plots.beeswarm(shap_values[:, :, 1])
shap.plots.bar(shap_values[:, :, 1])
shap.plots.force(shap_values[0, :, 1])
shap.plots.force(shap_values[:10, :, 1])
shap.plots.scatter(shap_values[:, "worst radius", 1])
shap.plots.heatmap(shap_values[:, :, 1])

# model = RandomForestClassifier(max_iter=1000)
clf_pipeline.fit(X_train, y_train)
rf_accuracy = clf_pipeline.score(X_test, y_test)
print(f"Model Accuracy Random : {rf_accuracy:.2%}")
