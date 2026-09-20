import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Cargar el dataset desde UCI
ruta = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'

columnas = [
    "mpg", "cylinders", "displacement", "horsepower", "weight",
    "acceleration", "model_year", "origin", "car_name"
]

auto = pd.read_csv(
    ruta,
    sep=r"\s+",
    names=columnas,         # Nombre asignado a las columnas
    na_values="?"           # Cadena asignada a valores sin datos
)

# Ver los primeros 5 filas del dataset
print(auto.head())

# 2. Inspeccion inicial
print("\nDimensiones:", auto.shape) # Saber las filas y columnas del dataset
print("\nValores faltanes:")
print(auto.isna().sum())

# display(auto.describe())

# 3. Eliminar los registros donde falten los caballos de fuerza (hp)
auto_simple = auto.dropna(subset=["horsepower"]).copy()
print("\nFilas disponibles:", len(auto_simple))

# 4. Preparar las variables
variables = [
    "weight",
    "horsepower",
    "displacement",
    "acceleration",
    "model_year",
    "cylinders"
]

datos_multiples = auto.dropna(subset=variables + ['mpg']).copy()

# 5. Entrenar el modelo

X_multiple = datos_multiples[variables]
y_multiple = datos_multiples['mpg']

modelo_multiple = LinearRegression()
modelo_multiple.fit(X_multiple, y_multiple)

print(f"Intercepto:", modelo_multiple.intercept_)
print(f"Coeficientes:")

for variable, coeficiente in zip(variables, modelo_multiple.coef_):
    print(f"{variable:15s}: {coeficiente:.4f}")

# 6. Evaluar la regresión lineal múltiple
y_pred_multiple = modelo_multiple.predict(X_multiple)

print(f"MAE : {mean_absolute_error(y_multiple, y_pred_multiple):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_multiple, y_pred_multiple)):.3f}")
print(f"R²  : {r2_score(y_multiple, y_pred_multiple):.3f}")

# 7. Valores reales vs Valores predichos
plt.scatter(y_multiple, y_pred_multiple, alpha=0.65, label="Predicciones")

limite_min = min(y_multiple.min(), y_pred_multiple.min())
limite_max = max(y_multiple.max(), y_pred_multiple.max())

plt.plot(
    [limite_min, limite_max],
    [limite_min, limite_max],
    linestyle="--",
    color="red",
    linewidth=2,
    label="Línea ideal"
)

plt.xlabel("MPG Real")
plt.ylabel("MPG Estimado")
# plt.title("Regresión Múltiple: Real vs Predicho")
plt.title("Regresión Múltiple: Todas las Variables → MPG ")
plt.legend()
plt.grid(True)
plt.show()