# REGRESION POLINOMIAL CON DATASET BIKESHARING (https://doi.org/10.24432/C5W894)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import zipfile, io, urllib.request
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Metodología: Cargar day.csv desde UCI
zip_url = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip" 
with urllib.request.urlopen(zip_url) as r:
    z = zipfile.ZipFile(io.BytesIO(r.read()))
    df = pd.read_csv(z.open("day.csv"))

# 2. Metodología: dteday -> cnt (días transcurridos)
df["dteday"] = pd.to_datetime(df["dteday"])
df = df.sort_values("dteday").reset_index(drop=True)
inicio = df["dteday"].min()
df["dateday"] = (df["dteday"] - inicio).dt.days

X = df[["dateday"]]
y = df["cnt"]
fechas = df["dteday"]

# 3. Metodología: Split 80/20
Xtr, Xte, ytr, yte, Ftr, Fte = train_test_split(X, y, fechas, test_size=0.2, random_state=42)

# 4. Resultados: Comparación de al menos 5 grados
grados_a_probar = [7] # Lista de los 5 escenarios

# Ver los primeros 10 registros del dataset general (ordenados)
print("--- PRIMEROS 20 REGISTROS GENERALES ---")
print(df[["dteday", "dateday", "cnt"]].head(10))


print("--- MÉTRICAS PARA REPORTE (RESULTADOS) ---")

for grado in grados_a_probar:
    # Entrenar el pipeline
    m = make_pipeline(PolynomialFeatures(grado, include_bias=False), LinearRegression())
    m.fit(Xtr, ytr)
    
    # Predicciones
    pred_train = m.predict(Xtr)
    pred_test = m.predict(Xte)
    
    # Calcular métricas para Train
    mae_train = mean_absolute_error(ytr, pred_train)
    rmse_train = np.sqrt(mean_squared_error(ytr, pred_train))
    r2_train = r2_score(ytr, pred_train)

    # Calcular métricas para Test
    rmse_test = np.sqrt(mean_squared_error(yte, pred_test))
    mae_test = mean_absolute_error(yte, pred_test)
    r2_test = r2_score(yte, pred_test)
    
    # Imprimir consola para la tabla del reporte
    print(f"\nGRADO {grado}:")
    print(f"TRAIN -> MAE: {mae_train:.2f} | RMSE: {rmse_train:.2f} | R2: {r2_train:.3f}")
    print(f"TEST  -> MAE: {mae_test:.2f} | RMSE: {rmse_test:.2f} | R2: {r2_test:.3f}")

    # Graficar
    plt.figure(figsize=(10, 6))
    plt.scatter(Ftr, ytr, alpha=0.3, label="Train (80%)", color='blue')
    plt.scatter(Fte, yte, alpha=0.7, label="Test (20%)", color='black')

    xs = np.linspace(0, df["dateday"].max(), 300).reshape(-1, 1)
    ys = m.predict(pd.DataFrame(xs, columns=["dateday"]))
    fechas_curva = [inicio + pd.Timedelta(days=int(v)) for v in xs.ravel()]

    plt.plot(fechas_curva, ys, label=f"Curva Polinomial (Grado {grado})", linewidth=3, color='red')
    plt.xlabel("Fecha (dteday)") 
    plt.ylabel("Alquileres (cnt)") 
    plt.title(f"Regresión Polinomial: Grado {grado}")
    plt.xticks(rotation=30) 
    plt.legend() 
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # Guarda 5 imágenes automáticamente
    plt.savefig(f"poli_grado_{grado}.png", dpi=150) 
    plt.close() # Cierra la figura en memoria para no saturar

print("\nSe han generado y guardado 5 gráficas exitosamente.")