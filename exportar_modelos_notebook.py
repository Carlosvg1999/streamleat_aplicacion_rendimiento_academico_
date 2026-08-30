"""Utilidades para exportar los modelos desde el Notebook del TFM."""

from pathlib import Path
import json

import joblib
import pandas as pd


def exportar_modelos(
    modelos: dict,
    X_referencia: pd.DataFrame,
    carpeta_salida="artefactos",
    descripciones_variables=None,
):
    """
    Guarda diez modelos y los metadatos que necesita la app.

    Los modelos XGBoost se guardan como UBJSON para evitar incompatibilidades
    entre sistemas operativos. Los demás modelos se conservan con joblib.

    Parameters
    ----------
    modelos:
        Diccionario {nombre visible: modelo entrenado}.
    X_referencia:
        DataFrame sin la variable objetivo. Debe tener las mismas columnas y
        codificación utilizadas para entrenar los modelos.
    carpeta_salida:
        Carpeta donde se crearán los modelos, el manifiesto y metadata.joblib.
    descripciones_variables:
        Diccionario opcional {nombre_columna: etiqueta comprensible}.
    """
    if len(modelos) != 10:
        raise ValueError("Debes proporcionar exactamente diez modelos.")
    if not isinstance(X_referencia, pd.DataFrame) or X_referencia.empty:
        raise ValueError("X_referencia debe ser un DataFrame no vacío.")
    if "riesgo" in X_referencia.columns:
        raise ValueError("X_referencia no debe contener la columna objetivo 'riesgo'.")

    carpeta = Path(carpeta_salida)
    carpeta.mkdir(parents=True, exist_ok=True)

    modelos_sklearn = {}
    manifiesto = []
    numero_xgboost = 0

    for nombre, modelo in modelos.items():
        # Si se ha guardado el objeto GridSearchCV completo, exportamos el mejor
        # estimador encontrado en lugar del buscador y todos sus resultados.
        modelo_final = getattr(modelo, "best_estimator_", modelo)
        modulo = modelo_final.__class__.__module__.lower()
        es_xgboost = (
            modulo.startswith("xgboost")
            and hasattr(modelo_final, "save_model")
        )

        if es_xgboost:
            numero_xgboost += 1
            archivo_modelo = f"xgboost_{numero_xgboost:02d}.ubj"
            modelo_final.save_model(carpeta / archivo_modelo)
            manifiesto.append(
                {
                    "nombre": str(nombre),
                    "tipo": "xgboost",
                    "archivo": archivo_modelo,
                }
            )
        else:
            modelos_sklearn[str(nombre)] = modelo_final
            manifiesto.append(
                {
                    "nombre": str(nombre),
                    "tipo": "sklearn",
                    "clave": str(nombre),
                }
            )

    if numero_xgboost != 5 or len(modelos_sklearn) != 5:
        raise ValueError(
            "Se esperaban cinco modelos XGBoost y cinco modelos de regresión "
            f"logística, pero se detectaron {numero_xgboost} y "
            f"{len(modelos_sklearn)}, respectivamente."
        )

    metadata = {
        "X_referencia": X_referencia.copy(),
        "nombres_clases": {
            0: "Mucho riesgo",
            1: "Riesgo",
            2: "Bajo riesgo",
        },
        "descripciones_variables": descripciones_variables or {},
    }

    joblib.dump(modelos_sklearn, carpeta / "modelos_sklearn.joblib")
    joblib.dump(metadata, carpeta / "metadata.joblib")
    with (carpeta / "modelos_manifest.json").open(
        "w",
        encoding="utf-8",
    ) as archivo:
        json.dump(manifiesto, archivo, ensure_ascii=False, indent=2)

    print(f"Archivos creados en: {carpeta.resolve()}")
    print("- modelos_sklearn.joblib")
    print("- modelos_manifest.json")
    print(f"- {numero_xgboost} modelos XGBoost en formato .ubj")
    print("- metadata.joblib")
