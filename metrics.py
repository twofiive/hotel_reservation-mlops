import joblib
import pandas as pd
from config.paths_config import *
from utils.common_functions import load_data
from sklearn.metrics import accuracy_score, f1_score

# Chargement modèle et données test
model = joblib.load(MODEL_OUTPUT_PATH)
test_df = load_data(PROCESSED_TEST_DATA_PATH)

X_test = test_df.drop(columns=['booking_status'])
y_test = test_df['booking_status']

y_pred = model.predict(X_test)

print(f'Accuracy  : {accuracy_score(y_test, y_pred):.4f}')
print(f'F1 Score  : {f1_score(y_test, y_pred):.4f}')
print(f'Modèle sauvegardé : {MODEL_OUTPUT_PATH}')
print(f'Features utilisées : {model.n_features_}')