"""
Ejecutar este script UNA VEZ para generar:
  - modelo_students.pkl
  - metricas_cv.pkl

Asegúrate de tener el CSV en datasets/StudentsPerformance.csv
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# ── 1. Cargar datos ───────────────────────────────────────────────────────────
df = pd.read_csv('datasets/StudentsPerformance.csv')

# ── 2. Encoding ───────────────────────────────────────────────────────────────
label_encoder = LabelEncoder()
categorical_cols = ['gender', 'race/ethnicity', 'parental level of education',
                    'lunch', 'test preparation course']
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])

X = df.drop('math score', axis=1)
y = df['math score']

# ── 3. RFE ────────────────────────────────────────────────────────────────────
rfe = RFE(estimator=RandomForestRegressor(n_estimators=100, random_state=42),
          n_features_to_select=4)
rfe.fit(X, y)

columnas_seleccionadas = X.columns[rfe.support_]
print("Features seleccionadas:", list(columnas_seleccionadas))
X_selected = X[columnas_seleccionadas]

# ── 4. Validación cruzada KFold=7 ─────────────────────────────────────────────
kf = KFold(n_splits=7, shuffle=True, random_state=42)
modelo_cv = RandomForestRegressor(n_estimators=100, random_state=42)

scores = cross_validate(
    estimator=modelo_cv,
    X=X_selected, y=y,
    cv=kf,
    scoring={"R2": "r2", "MAE": "neg_mean_absolute_error"}
)

r2_corridas  = np.round(scores["test_R2"], 4).tolist()
mae_corridas = np.round(-scores["test_MAE"], 4).tolist()

metricas = {
    "r2_corridas":  r2_corridas,
    "mae_corridas": mae_corridas,
    "r2_promedio":  round(float(np.mean(r2_corridas)),  4),
    "mae_promedio": round(float(np.mean(mae_corridas)), 4),
}

print("Métricas CV guardadas:", metricas)

# ── 5. Modelo final (train/test split) ────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42)

modelo_final = RandomForestRegressor(n_estimators=100, random_state=42)
modelo_final.fit(X_train, y_train)
y_pred = modelo_final.predict(X_test)

metricas["r2_individual"]  = round(float(r2_score(y_test, y_pred)),          4)
metricas["mae_individual"] = round(float(mean_absolute_error(y_test, y_pred)), 4)

print(f"R² individual: {metricas['r2_individual']}")
print(f"MAE individual: {metricas['mae_individual']}")

# ── 6. Guardar ────────────────────────────────────────────────────────────────
metricas["features"] = list(columnas_seleccionadas)

joblib.dump(modelo_final, 'modelo_students.pkl')
joblib.dump(metricas,     'metricas_cv.pkl')
print("✅ modelo_students.pkl y metricas_cv.pkl guardados.")
print("Features guardadas:", metricas["features"])
