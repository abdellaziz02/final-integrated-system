# train_hybrid_model.py (Final Tuned Version)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import joblib

df = pd.read_csv('supplier_engineered_features.csv')
X = df.drop(['url', 'label'], axis=1); y = df['label']
text_feature = 'key_text'; numeric_features = [col for col in X.columns if col != text_feature]

preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(stop_words='english', ngram_range=(1,2)), text_feature),
        ('numeric', StandardScaler(), numeric_features)
    ],
    remainder='passthrough'
)

# --- UPGRADE: Tuned XGBoost Hyperparameters ---
# These parameters make the model more powerful and less prone to simple mistakes.
xgb_tuned = XGBClassifier(
    use_label_encoder=False,
    eval_metric='mlogloss',
    n_estimators=200,      # More trees
    learning_rate=0.1,     # Slower learning
    max_depth=5,           # Deeper trees
    subsample=0.8,         # Use 80% of data per tree
    colsample_bytree=0.8,  # Use 80% of features per tree
    random_state=42
)

model_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', xgb_tuned)])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
print("Training Final Tuned Hybrid Model...")
model_pipeline.fit(X_train, y_train)

print("\nEvaluating Final Tuned Model...")
y_pred = model_pipeline.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['B2B', 'Producer', 'Retailer', 'Info/Blog']))

joblib.dump(model_pipeline, 'hybrid_supplier_model.joblib')
joblib.dump(list(X.columns), 'model_columns.joblib')
print("\nFinal hybrid model trained and saved successfully!")