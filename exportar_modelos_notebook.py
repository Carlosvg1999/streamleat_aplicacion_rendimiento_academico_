"""Utilidades para exportar los modelos desde el Notebook del TFM."""

from pathlib import Path

import joblib
import pandas as pd


def exportar_modelos(
    modelos: dict,
    X_referencia: pd.DataFrame,
    carpeta_salida="artefactos",
    descripciones_variables=None,
):
    """
    Guarda los cuatro modelos y los metadatos que necesita la app.

    Parameters
    ----------
    modelos:
        Diccionario {nombre visible: modelo entrenado}.
    X_referencia:
        DataFrame sin la variable objetivo. Debe tener las mismas columnas y
        codificación utilizadas para entrenar los modelos.
    carpeta_salida:
        Carpeta donde se crearán modelos.joblib y metadata.joblib.
    descripciones_variables:
        Diccionario opcional {nombre_columna: etiqueta comprensible}.
    """
    if len(modelos) != 4:
        raise ValueError("Debes proporcionar exactamente cuatro modelos.")
    if not isinstance(X_referencia, pd.DataFrame) or X_referencia.empty:
        raise ValueError("X_referencia debe ser un DataFrame no vacío.")
    if "riesgo" in X_referencia.columns:
        raise ValueError("X_referencia no debe contener la columna objetivo 'riesgo'.")

    carpeta = Path(carpeta_salida)
    carpeta.mkdir(parents=True, exist_ok=True)

    metadata = {
        "X_referencia": X_referencia.copy(),
        "nombres_clases": {
            0: "Mucho riesgo",
            1: "Riesgo",
            2: "Bajo riesgo",
        },
        "descripciones_variables": descripciones_variables or {},
    }

    joblib.dump(modelos, carpeta / "modelos.joblib")
    joblib.dump(metadata, carpeta / "metadata.joblib")

    print(f"Archivos creados en: {carpeta.resolve()}")
    print("- modelos.joblib")
    print("- metadata.joblib")
