"""Celda para ejecutar en Colab después de entrenar los diez modelos."""

from google.colab import files

from exportar_modelos_notebook import exportar_modelos


modelos_streamlit = {
    "Regresión logística — Experimento 1": modelo_logistico_1,
    "XGBoost — Experimento 1": modelo_XGBoost_1,
    "Regresión logística — Experimento 2": modelo_logistico_2,
    "XGBoost — Experimento 2": modelo_XGBoost_2,
    "Regresión logística — Experimento 3": modelo_logistico_3,
    "XGBoost — Experimento 3": modelo_XGBoost_3,
    "Regresión logística — Experimento 4": modelo_logistico_4,
    "XGBoost — Experimento 4": modelo_XGBoost_4,
    "Regresión logística — Experimento 5": modelo_logistico_5,
    "XGBoost — Experimento 5": modelo_XGBoost_5,
}


X_referencia = (
    df_modelo_test
    .drop(columns="riesgo")
    .copy()
    .reset_index(drop=True)
)


descripciones_variables = {
    "famsize": "Tamaño familiar",
    "Pstatus": "Estado civil de los padres",
    "educacion_familiar": "Educación familiar",
    "apoyo_familiar": "Apoyo familiar",
    "studytime": "Tiempo de estudio semanal",
    "failures": "Número de suspensos previos",
    "schoolsup": "Apoyo educativo adicional",
    "activities": "Actividades extraescolares",
    "higher": "Intención de estudiar cursos superiores",
    "famrel": "Calidad de las relaciones familiares",
    "freetime": "Tiempo libre después de la escuela",
    "goout": "Salidas con amigos",
}


ruta_zip = exportar_modelos(
    modelos=modelos_streamlit,
    X_referencia=X_referencia,
    carpeta_salida="/content/artefactos",
    descripciones_variables=descripciones_variables,
)


files.download(str(ruta_zip))
