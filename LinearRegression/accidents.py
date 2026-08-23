import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Cargar el dataset
ruta = '../datasets/accidents.csv'          # Ruta de los datasets
df = pd.read_csv(ruta)

# 2. Definir los nombres de las columnas a analizar
columna_x = 'x'         # Numero de accientes de transito de cada estado
columna_y = 'y'         # Numero de fatalidades en ese estado

# 3. Visualizar los datos originales
plt.scatter(df[columna_x], df[columna_y], s=75, color='blue', alpha=0.6)
plt.xlabel("Accidentes de tránsito")
plt.ylabel("Fatalidades")
plt.title(f"Accidentes de tránsito vs Fatalidades")
plt.grid(True)
plt.show()

# Definir X e Y para SciKit Learn
X = df[[columna_x]]
y = df[columna_y]

# 4. Crear y entrenar el modelo
modelo = LinearRegression()
modelo.fit(X, y)

print()
print("Intercepto:", modelo.intercept_)
print("Pendiente:", modelo.coef_[0])
print()

# 5. Generar recta de regresion lineal
x_linea_np = np.linspace(X[columna_x].min(), X[columna_x].max(), 100).reshape(-1, 1)
x_linea_df = pd.DataFrame(x_linea_np, columns=[columna_x])

y_linea = modelo.predict(x_linea_df)

# 6. Visualizar grafica con regresion lineal
plt.scatter(X, y, s=75, color='blue', alpha=0.6, label="Datos Observados")
plt.plot(x_linea_df, y_linea, linewidth=2, color='red', label='Regresión Lineal')
plt.xlabel("Accidentes")
plt.ylabel("Fatalidades")
plt.title(f"Regresión Lineal: Accidentes de Tránsito → Fatalidades")
plt.legend()
plt.grid(True)
plt.show()

# 7. Evaluar el modelo
y_pred = modelo.predict(X)

mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

print()
print("-" * 20)
print("MÉTRICAS DE EVALUACIÓN")
print(f"MAE : {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R²  : {r2:.3f}")
print()

# 8.Predecir una nueva evaluación

# Crear dataframe
acc_hipoteticos = 500000
nw_observacion = pd.DataFrame({columna_x: [acc_hipoteticos]})

# Hacer prediccion
pred_fatalidad = modelo.predict(nw_observacion)

# Mostrar resultado
print(f"En {acc_hipoteticos:,.0f} accidentes de tránsito se tendrán: {pred_fatalidad[0]:.2f} fatalidades")