import joblib
import pandas as pd
import shap
from imblearn.over_sampling import SMOTE
from plain_english import explain_in_plain_english
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

print("Reading data...")

df = pd.read_csv("creditcard.csv")
features = [f"V{i}" for i in range(1, 29)]
X = df[features]
y = df["Class"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


print("Data read successfully.")


param_dict = {"n_estimators": [100, 200, 300], "max_depth": [10, 20, 30, None]}


print("Performing grid search...")

# grid_search = GridSearchCV(
#     estimator = RandomForestClassifier(),
#     param_grid=param_dict,
#     cv=5,
# )
# grid_search.fit(X_train, y_train)

print("Grid search completed.")

# print(grid_search.best_params_)

print("Fitting model...")
clf_pipeline = RandomForestClassifier()
clf_pipeline.fit(X_train, y_train)
print("Model fitted successfully.")

print("Applying SMOTE...")
smote = SMOTE(sampling_strategy="auto", random_state=42, k_neighbors=5)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print("SMOTE applied successfully.")

print("Defining preprocessor...")
numerical_col = [
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "V7",
    "V8",
    "V9",
    "V10",
    "V11",
    "V12",
    "V13",
    "V14",
    "V15",
    "V16",
    "V17",
    "V18",
    "V19",
    "V20",
    "V21",
    "V22",
    "V23",
    "V24",
    "V25",
    "V26",
    "V27",
    "V28",
]
numerical_transformer = Pipeline(
    steps=[("input", SimpleImputer(strategy="median")), ("scalar", StandardScaler())]
)
preprocessor = ColumnTransformer(
    transformers=[("num", StandardScaler(), numerical_col)]
)
clf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("rf_model", RandomForestClassifier(random_state=42)),
    ]
)
print("Preprocessor defined successfully.")
print("Fitting model...")
clf_pipeline.fit(X_train_sm, y_train_sm)
print("Model fitted successfully.")
print("Making predictions...")
y_pred = clf_pipeline.predict(X_test)
print("Predictions made successfully.")
print("Classification report:")
print(classification_report(y_test, y_pred))

explainer = shap.TreeExplainer(clf_pipeline)
shap_values = explainer(X_test)
print(shap_values)

shap.plots.waterfall(shap_values[0, :, 1])  # your existing line
explain_in_plain_english(shap_values[0, :, 1])

# model = RandomForestClassifier(max_iter=1000)
clf_pipeline.fit(X_train, y_train)
print("Model fitted successfully.")
print("Calculating accuracy...")
rf_accuracy = clf_pipeline.score(X_test, y_test)
print(f"Model Accuracy Random : {rf_accuracy:.2%}")
print("Accuracy calculated successfully.")
joblib.dump(clf_pipeline, "clf_pipeline.pkl")
print("Done! Never Train again")