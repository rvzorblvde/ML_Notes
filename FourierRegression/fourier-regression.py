import pandas as pd, matplotlib.pyplot as plt, numpy as np
import zipfile, io, urllib.request
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Cargar Dataset BikeSharing desde la URL oficial
zip_url = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
with urllib.request.urlopen(zip_url) as r:
    df = pd.read_csv(zipfile.ZipFile(io.BytesIO(r.read())).open("day.csv"))

# 2. Variables de tiempo
df["dteday"] = pd.to_datetime(df["dteday"])
inicio = df["dteday"].min()
df["dateday"] = (df["dteday"] - inicio).dt.days
PERIODO = 365.25 # El dataset presenta un periodo anual (365 días)

# Ver los primeros 10 registros del dataset general (ordenados)
print("--- PRIMEROS 10 REGISTROS GENERALES ---")
print(df[["dteday", "dateday", "cnt"]].head(10))

# Función para generar características de Fourier
def fourier(t, n_armonicos, periodo=PERIODO):
    t = np.asarray(t).ravel()
    cols = {}
    for k in range(1, n_armonicos + 1):
        cols[f"sin{k}"] = np.sin(2 * np.pi * k * t / periodo)
        cols[f"cos{k}"] = np.cos(2 * np.pi * k * t / periodo)
    return pd.DataFrame(cols)

# Variables objetivo y arreglos base para las gráficas
y = df["cnt"]
fechas = df["dteday"]
t_num = df["dateday"]

xs = np.linspace(0, df["dateday"].max(), 500)
xs_fechas = [inicio + pd.Timedelta(days=float(v)) for v in xs]

# 3. Metodología: Split 80/20 sobre la variable temporal para poder calcular K dinámicamente
Xtr_t, Xte_t, ytr, yte, Ftr, Fte = train_test_split(t_num, y, fechas, test_size=0.2, random_state=42)

# Valores de K a evaluar y paleta de colores para diferenciarlos
k_valores = [2, 5, 8, 15, 26]
colores = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231']

# Configuración de gráfica maestra
fig_maestra = plt.figure(figsize=(12, 7))
plt.scatter(Fte, yte, alpha=0.4, label="Test real", color='black')

# Encabezado de la tabla de métricas (Train y Test)
print("-" * 88)
print(f"{'K':<4} | {'Train RMSE':<11} | {'Train MAE':<10} | {'Train R2':<9} | {'Test RMSE':<10} | {'Test MAE':<9} | {'Test R2':<8}")
print("-" * 88)

for i, k in enumerate(k_valores):
    # Generar matriz de diseño (features) para el K actual
    Xtr = fourier(Xtr_t, k)
    Xte = fourier(Xte_t, k)
    X_plot = fourier(xs, k)
    
    # Entrenar modelo
    m = LinearRegression().fit(Xtr, ytr)
    
    # Predicciones
    pred_tr = m.predict(Xtr)
    pred_te = m.predict(Xte)
    y_curve = m.predict(X_plot)
    
    # Métricas Training
    rmse_tr = np.sqrt(mean_squared_error(ytr, pred_tr))
    mae_tr = mean_absolute_error(ytr, pred_tr)
    r2_tr = r2_score(ytr, pred_tr)

    # Métricas Test
    rmse_te = np.sqrt(mean_squared_error(yte, pred_te))
    mae_te = mean_absolute_error(yte, pred_te)
    r2_te = r2_score(yte, pred_te)
    
    # Imprimir resultados en consola
    print(f"{k:<4} | {rmse_tr:<11.2f} | {mae_tr:<10.2f} | {r2_tr:<9.3f} | {rmse_te:<10.2f} | {mae_te:<9.2f} | {r2_te:<8.3f}")
    
    # --- 1. AÑADIR A LA GRÁFICA MAESTRA ---
    plt.figure(fig_maestra.number) 
    plt.plot(xs_fechas, y_curve, label=f"K={k}", color=colores[i], linewidth=2)
    
    # --- 2. CREAR Y GUARDAR LA GRÁFICA INDIVIDUAL ---
    fig_indiv = plt.figure(figsize=(10, 6))
    plt.scatter(Fte, yte, alpha=0.4, label="Test", color='black')
    plt.plot(xs_fechas, y_curve, label=f"Fourier K={k}", color=colores[i], linewidth=2)
    
    plt.xlabel("Fecha (dteday)") 
    plt.ylabel("Alquileres (cnt)") 
    plt.title(f"Regresión con Fourier (Armónico K={k})\nTrain R2: {r2_tr:.3f} | Test R2: {r2_te:.3f}")
    plt.xticks(rotation=30) 
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"fourier_individual_K{k}.png", dpi=300) 
    plt.close(fig_indiv)

print("-" * 88)

# --- FINALIZAR Y MOSTRAR LA GRÁFICA MAESTRA ---
plt.figure(fig_maestra.number)
plt.xlabel("Fecha (dteday)") 
plt.ylabel("Alquileres (cnt)") 
plt.title("Comparación de Regresión con Fourier (Múltiples Armónicos)")
plt.xticks(rotation=30) 
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
plt.tight_layout()
plt.savefig("fourier_comparacion_todos.png", dpi=300) 
plt.show()