import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib

# 1. Cargar dataset de Zetas
files = glob.glob("imagenes_piezas/Zeta/*_features.csv")
df_list = [pd.read_csv(f, header=None) for f in files]
df = pd.concat(df_list, ignore_index=True)

# 2. Asignar nombres de columnas
df.columns = [
    "piece_id","Roto","col_1","col_2","col_3","col_4","col_5","col_6",
    "Hu_1","Hu_2","Hu_3","Hu_4","Hu_5","Hu_6","Hu_7"
]

# 3. Etiqueta: 1 = Zeta real, 0 = Defectuosa
df["label"] = df["Roto"]

# 4. Features y etiquetas
X = df[["col_1","col_2","col_3","col_4","col_5","col_6",
        "Hu_1","Hu_2","Hu_3","Hu_4","Hu_5","Hu_6","Hu_7"]]
y = df["label"]

# 5. Dividir dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Entrenar modelo SVM con class_weight="balanced"
svm_model = SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced", probability=True)
svm_model.fit(X_train, y_train)

# 7. Evaluar
y_pred = svm_model.predict(X_test)
print("Resultados SVM balanceado:")
print(classification_report(y_test, y_pred))

# 8. Guardar modelo
joblib.dump(svm_model, "zeta_svm_model.pkl")