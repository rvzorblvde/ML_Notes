import pandas as pd, matplotlib.pyplot as plt, numpy as np
import zipfile, io, urllib.request
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import SplineTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Cargar el dataset de Bike Sharing 
zip_url = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
with urllib.request.urlopen(zip_url) as r:
    df = pd.read_csv(zipfile.ZipFile(io.BytesIO(r.read())).open("day.csv"))

# Variables: dateday (fecha) -> cnt (alquileres). t = dias desde el inicio.
df["dteday"] = pd.to_datetime(df["dteday"])
inicio = df["dteday"].min()
df["dateday"] = (df["dteday"] - inicio).dt.days # t numérico

X = df[["dateday"]]
y = df["cnt"]
fechas = df["dteday"]

# 2. Metodologia: Split 80/20
Xtr, Xte, ytr, yte, Ftr, Fte = train_test_split(X, y, fechas, test_size=0.2, random_state=42)

# 3. Resultados
nudos_test = [5, 8, 12, 16, 22]

# Puntos X para dibujar las curvas suavizadas
xs = np.linspace(0, df["dateday"].max(), 300).reshape(-1, 1)
xs_df = pd.DataFrame(xs, columns=["dateday"])
xs_fechas = [inicio + pd.Timedelta(days=int(v)) for v in xs.ravel()]

# --- CONFIGURAR LA GRÁFICA MAESTRA (TODOS LOS MODELOS) ---
fig_maestra = plt.figure(figsize=(12, 7))
plt.scatter(Fte, yte, alpha=0.3, label="Test", color='blue')

# Tablas de métricas
print("-" * 50)
print(f"{'Nudos':<8} | {'RMSE':<10} | {'MAE':<10} | {'R2':<10}")
print("-" * 50)

# Bucle para entrenar y graficar
for nudos in nudos_test:
    m = make_pipeline(
        SplineTransformer(degree=3, n_knots=nudos, knots="quantile", include_bias=False),
        LinearRegression()
    )
    
    m.fit(Xtr, ytr)
    pred = m.predict(Xte)
    
    # Cálculo de métricas
    rmse = np.sqrt(mean_squared_error(yte, pred))
    mae = mean_absolute_error(yte, pred)
    r2 = r2_score(yte, pred)
    
    # Imprimir fila en la tabla de la consola
    print(f"{nudos:<8} | {rmse:<10.2f} | {mae:<10.2f} | {r2:<10.3f}")
    
    # Generar la curva de predicción para este modelo
    y_curve = m.predict(xs_df)
    
    # Añadir a la grafica final
    # Volvemos a activar la figura maestra por si se perdió el foco
    plt.figure(fig_maestra.number) 
    plt.plot(xs_fechas, y_curve, label=f"{nudos} nudos", linewidth=2)
    
    # Crear imagen individual
    fig_indiv = plt.figure(figsize=(10, 6))
    plt.scatter(Fte, yte, alpha=0.3, label="Test", color='blue')
    plt.plot(xs_fechas, y_curve, label=f"Spline {nudos} nudos", color='red', linewidth=2)
    
    plt.xlabel("Fecha (dteday)") 
    plt.ylabel("Alquileres (cnt)") 
    plt.title(f"Regresión Spline Cúbica - {nudos} Nudos\nRMSE: {rmse:.2f} | R2: {r2:.3f}")
    plt.xticks(rotation=30) 
    plt.legend()
    plt.tight_layout()
    
    # Guardar cada imagen 
    plt.savefig(f"spline_individual_{nudos}_nudos.png", dpi=300) 
    plt.close(fig_indiv)

print("-" * 50)

# Mostrar grafica con todos los nudos
plt.figure(fig_maestra.number)
plt.xlabel("Fecha (dteday)") 
plt.ylabel("Alquileres (cnt)") 
plt.title("Comparación de Regresión Spline Cúbica con diferentes nudos")
plt.xticks(rotation=30) 
plt.legend()
plt.tight_layout()
plt.savefig("splines_comparacion_todos.png", dpi=300) 
plt.show()