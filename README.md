# Aplicación Streamlit de riesgo académico

La aplicación permite introducir los datos de un alumno y devuelve:

- las probabilidades de pertenencia a las clases 0, 1 y 2 para cuatro modelos;
- cuatro gráficas comparables;
- la clase estimada por cada modelo;
- una estimación conjunta calculada como promedio simple de probabilidades;
- las tres variables más relevantes para cada predicción mediante SHAP.

## 1. Exportar los modelos desde el Notebook

Copia `exportar_modelos_notebook.py` al entorno donde tienes los modelos o
ejecuta su contenido como una celda. Después utiliza:

```python
from exportar_modelos_notebook import exportar_modelos

modelos_streamlit = {
    "XGBoost — Experimento 5": modelo_XGBoost_5,
    "Regresión logística — Experimento 5": modelo_logistico_5,
    "XGBoost — Experimento 4": modelo_XGBoost_4,
    "Regresión logística — Experimento 4": modelo_logistico_4,
}

X_referencia = df_modelo_test.drop(columns="riesgo")

exportar_modelos(
    modelos=modelos_streamlit,
    X_referencia=X_referencia,
    carpeta_salida="artefactos",
)
```

Si alguno de tus modelos tiene otro nombre de variable, cambia únicamente el
valor situado a la derecha de los dos puntos.

Para que el formulario use textos más comprensibles puedes añadir descripciones:

```python
descripciones = {
    "studytime": "Tiempo semanal de estudio",
    "failures": "Número de suspensos previos",
    "absences": "Número de ausencias",
}

exportar_modelos(
    modelos=modelos_streamlit,
    X_referencia=X_referencia,
    carpeta_salida="artefactos",
    descripciones_variables=descripciones,
)
```

Los archivos resultantes deben quedar aquí:

```text
streamlit_riesgo_academico/
├── app.py
├── requirements.txt
├── exportar_modelos_notebook.py
└── artefactos/
    ├── modelos.joblib
    └── metadata.joblib
```

No uses como referencia datos sintéticos ni incluyas la columna `riesgo`. Lo
más coherente con la evaluación del TFM es emplear las variables predictoras del
conjunto real de prueba o, preferentemente, una muestra real reservada para
referencia que no exponga información personal identificable.

## 2. Instalar y ejecutar

Desde una terminal situada en esta carpeta:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Streamlit mostrará una dirección local, normalmente
`http://localhost:8501`.

## 3. Consideraciones importantes

- Los cuatro modelos deben aceptar las mismas variables que aparecen en
  `X_referencia` y deben disponer de `predict_proba`.
- La aplicación espera las clases numéricas 0, 1 y 2.
- Los modelos y el conjunto de referencia deben conservar exactamente la misma
  codificación y nombres de columnas utilizados durante el entrenamiento.
- SHAP explica cómo se comporta el modelo, pero no demuestra causalidad.
- Las probabilidades estimadas no equivalen necesariamente a probabilidades
  calibradas. Para presentarlas como riesgo probabilístico conviene evaluar la
  calibración de cada modelo.
- El resultado conjunto es una agregación descriptiva. No sustituye la
  validación de un ensemble entrenado y evaluado formalmente.

## 4. Despliegue

Para desplegar la aplicación en Streamlit Community Cloud, sube la carpeta a
un repositorio privado y selecciona `app.py` como archivo principal. Si los
modelos o los datos contienen información sensible, no utilices un repositorio
público y revisa las obligaciones de protección de datos aplicables.
