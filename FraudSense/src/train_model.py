import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load dataset
df = pd.read_csv('data/fraud_dataset.csv')

# Drop unnecessary cols
df_model = df.drop(['nameOrig', 'nameDest', 'isFlaggedFraud', 'step'], axis=1)

# Features and target
X = df_model.drop('isFraud', axis=1)
y = df_model['isFraud']

# Define numeric & categorical features
numeric = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
categorical = ['type']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric),
        ('cat', OneHotEncoder(drop='first'), categorical)
    ]
)

# Pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=500, class_weight='balanced'))
])

# Train
pipeline.fit(X_train, y_train)

# Evaluation
y_pred = pipeline.predict(X_test)
print('Classification Report:\n', classification_report(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(pipeline, 'models/fraud_detection_pipeline.pkl')
print('✅ Model saved to models/fraud_detection_pipeline.pkl')
