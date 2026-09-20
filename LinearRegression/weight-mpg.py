import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Cargar el dataset desde UCI
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

# ¿Podemos estimar el consumo de un automovil a partir de su peso?

# 4. Visualizar los datos
plt.scatter(auto_simple["weight"], auto_simple["mpg"], alpha=0.65)
plt.xlabel("Peso")
plt.ylabel("Millas por Galón")
plt.title(f"Auto: Peso vs MPG")
plt.grid(True)
plt.show()

# 5. Entrenar el modelo
X = auto_simple[["weight"]]
y = auto_simple["mpg"]

modelo_mpg = LinearRegression()
modelo_mpg.fit(X, y)

print("Intercepto:", modelo_mpg.intercept_)
print("Pendiente:", modelo_mpg.coef_[0])

# 6. Graficar recta de regresión
x_linea = np.linspace(X["weight"].min(), X["weight"].max(), 200).reshape(-1, 1)
y_linea = modelo_mpg.predict(x_linea)

plt.scatter(X["weight"], y, alpha=0.65, label="Datos")
plt.plot(x_linea, y_linea, linewidth=2, color='red', label='Recta de Regresión')
plt.xlabel("Peso")
plt.ylabel("Millas Por Galón")
plt.title(f"Regresión Lineal Simple: Peso → MPG")
plt.legend()
plt.grid(True)
plt.show()

# 7. Evaluar el modelo
y_pred = modelo_mpg.predict(X)

print(f"MAE : {mean_absolute_error(y, y_pred):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y, y_pred)):.3f}")
print(f"R²  : {r2_score(y, y_pred):.3f}")

# Predicción
auto_nuevo = pd.DataFrame({"weight": [3000]})
mpg_estimado = modelo_mpg.predict(auto_nuevo)
print(f"MPG estimadas: {mpg_estimado[0]:.2f}")