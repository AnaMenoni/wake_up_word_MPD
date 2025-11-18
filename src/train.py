from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

def train_model(X, y, model_path="modelo.pkl"):
    # Directorio raíz del proyecto (para guardar modelo y scaler siempre en el mismo lugar)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_out = os.path.join(BASE_DIR, model_path)
    scaler_out = os.path.join(BASE_DIR, "scaler.pkl")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # === SCALING ===
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # === MODEL ===
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== Resultados del entrenamiento ===")
    print(f"Precisión: {acc:.3f}")
    print("Matriz de confusión:")
    print(cm)

    # === GUARDAR ===
    joblib.dump(model, model_out)
    joblib.dump(scaler, scaler_out)

    print(f"\nModelo guardado en {model_out}")
    print(f"Scaler guardado en {scaler_out}")
