import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import joblib


def main() -> None:
    df = pd.read_csv("./imagenes_piezas/Anillo/Anillo_features.csv", header=None)
    y = df.iloc[:, 1]
    x = df.iloc[:, 2:]

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    pca = PCA(n_components=5)
    x_scaled_pca = pca.fit_transform(x_scaled)

    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled_pca, y, test_size=0.3, random_state=42
    )

    model_pca = SVC(kernel="rbf", C=1.0)
    model_pca.fit(x_train, y_train)
    y_predict = model_pca.predict(x_test)

    # --- Add the saving step here ---
    model_filename = "svc_model_anillos.joblib"
    scaler_filename = "scaler_anillos.joblib"
    pca_filename = "pca_anillos.joblib"

    joblib.dump(model_pca, model_filename)
    joblib.dump(scaler, scaler_filename)
    joblib.dump(pca, pca_filename)

    print(f"Model saved to: {model_filename}")
    print(f"Scaler saved to: {scaler_filename}")
    print(f"PCA saved to: {pca_filename}")

    print(f"{accuracy_score(y_test,y_predict)=}")
    print(classification_report(y_test, y_predict, zero_division=0))


if __name__ == "__main__":
    main()
