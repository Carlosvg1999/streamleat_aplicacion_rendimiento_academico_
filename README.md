# Aplicación de estimación del riesgo académico

Aplicación desarrollada con Streamlit para comparar diez modelos de clasificación
del riesgo académico. Forma parte de un Trabajo Fin de Máster sobre el uso de datos
reales y sintéticos en modelos educativos.

La herramienta constituye un apoyo para el análisis y no debe utilizarse como único
criterio para tomar decisiones que afecten al alumnado.

## Funcionalidades

- Formulario con valores y descripciones comprensibles para el usuario.
- Comparación de cinco regresiones logísticas y cinco modelos XGBoost.
- Gráfica del grado de pertenencia a cada una de las tres clases para cada modelo.
- Clase con mayor probabilidad y estimación conjunta de los diez modelos.
- Explicación local en lenguaje natural de las tres variables más influyentes según
  SHAP para la clase escogida por cada modelo.
- Presentación adaptable a ordenadores, tabletas y teléfonos móviles.

Las clases utilizadas son:

| Clase | Denominación | Calificación final |
|---|---|---|
| `0` | Mucho riesgo | Inferior a 8 |
| `1` | Riesgo | Entre 8 y 12 |
| `2` | Bajo riesgo | Superior a 12 |

## Estructura del proyecto

```text
streamlit_riesgo_academico/
├── app.py
├── requirements.txt
├── README.md
├── exportar_modelos_notebook.py
├── celda_exportacion_colab.py
├── .streamlit/
│   └── config.toml
└── artefactos/
    ├── metadata.joblib
    ├── modelos_sklearn.joblib
    ├── modelos_manifest.json
    ├── xgboost_01.ubj
    ├── xgboost_02.ubj
    ├── xgboost_03.ubj
    ├── xgboost_04.ubj
    └── xgboost_05.ubj
```

Los archivos deben estar directamente dentro de `artefactos`. Una estructura como
`artefactos/artefactos/metadata.joblib` no será reconocida por la aplicación.

## Exportación de los modelos desde Colab

XGBoost no garantiza la portabilidad entre entornos de los objetos completos
guardados mediante `pickle` o `joblib`. Por ello:

- las cinco regresiones logísticas se guardan en `modelos_sklearn.joblib`;
- los cinco modelos XGBoost se guardan individualmente en formato UBJSON (`.ubj`);
- `modelos_manifest.json` conserva el nombre, tipo y orden de los diez modelos;
- `metadata.joblib` contiene las variables de referencia, nombres de clases y
  versiones del entorno de entrenamiento.

### 1. Preparar el código

Ejecuta en Colab el contenido de `exportar_modelos_notebook.py`. Después ejecuta
`celda_exportacion_colab.py`, que utiliza los modelos ya entrenados:

```python
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
```

Las variables de referencia se obtienen sin incluir la variable objetivo:

```python
X_referencia = (
    df_modelo_test
    .drop(columns="riesgo")
    .copy()
    .reset_index(drop=True)
)
```

### 2. Exportar y verificar

```python
ruta_zip = exportar_modelos(
    modelos=modelos_streamlit,
    X_referencia=X_referencia,
    carpeta_salida="/content/artefactos",
    descripciones_variables=descripciones_variables,
)
```

El exportador realiza las siguientes comprobaciones antes de generar el ZIP:

- existen exactamente diez modelos;
- se detectan cinco XGBoost y cinco modelos de regresión logística;
- todos disponen de `predict_proba()`;
- todos utilizan las clases `0`, `1` y `2`;
- el número de variables coincide con `X_referencia`;
- los modelos recargados producen las mismas probabilidades que los originales.

Si recibe un objeto `GridSearchCV`, utiliza automáticamente su `best_estimator_`.

### 3. Descargar los artefactos

```python
from google.colab import files

files.download(str(ruta_zip))
```

Descomprime `artefactos.zip` y copia todos sus archivos en la carpeta `artefactos`
del proyecto.

No se debe incluir la columna `riesgo` ni información identificativa del alumnado en
`X_referencia`. Para este proyecto se utiliza el conjunto real reservado como
referencia, no los registros sintéticos empleados para entrenar algunos experimentos.

## Valores interpretables del formulario

El formulario muestra textos comprensibles, pero conserva internamente los códigos
numéricos que reciben los modelos. Por ejemplo:

- `studytime = 3` se presenta como «Entre 5 y 10 horas»;
- `failures = 0` se presenta como «Ninguno»;
- `schoolsup = 1` se presenta como «Sí recibe apoyo»;
- `famrel = 5` se presenta como «Excelente».

Las explicaciones SHAP utilizan esas mismas denominaciones. Debe comprobarse que las
codificaciones de `famsize`, `Pstatus`, `educacion_familiar` y `apoyo_familiar`
coinciden con las transformaciones aplicadas en el Notebook. Cambiar el texto visible
no modifica el valor numérico enviado al modelo.

## Instalación local

Se recomienda utilizar Python 3.13 y un entorno virtual.

### Windows y PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Si PowerShell bloquea la activación:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### macOS o Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Ejecutar la aplicación

Desde la carpeta que contiene `app.py`:

```powershell
python -m streamlit run app.py
```

La aplicación estará normalmente disponible en:

```text
http://localhost:8501
```

Para reiniciarla:

```text
1. Pulsa Ctrl + C en la terminal.
2. Ejecuta: python -m streamlit cache clear
3. Ejecuta: python -m streamlit run app.py
```

## Actualizar la aplicación

Después de modificar `app.py`, comprueba primero su funcionamiento local. A
continuación, sube los cambios al repositorio:

```powershell
git status
git diff
git add app.py README.md requirements.txt
git commit -m "Actualizada la aplicación de riesgo académico"
git pull --rebase origin main
git push origin main
```

Si se vuelven a entrenar los modelos, genera nuevamente `artefactos.zip`, sustituye
todos los archivos de `artefactos` y ejecuta:

```powershell
git add artefactos/
git commit -m "Actualizados los modelos predictivos"
git pull --rebase origin main
git push origin main
```

Streamlit Community Cloud detectará el nuevo commit y reconstruirá la aplicación.

## Despliegue en Streamlit Community Cloud

Configura el despliegue con:

| Campo | Valor |
|---|---|
| Repositorio | Repositorio de GitHub que contiene el proyecto |
| Rama | `main` |
| Archivo principal | `app.py` |

El repositorio debe contener `app.py`, `requirements.txt` y todos los artefactos. Si
el repositorio es privado, Streamlit debe estar autorizado para acceder a él.

## Interpretación responsable

- Las salidas representan estimaciones de los modelos, no diagnósticos.
- SHAP explica la contribución local de las variables a una predicción, pero no
  demuestra relaciones causales.
- Las probabilidades no deben presentarse como calibradas si no se ha evaluado su
  calibración expresamente.
- El promedio de los diez modelos es un resumen descriptivo, no un *ensemble*
  entrenado y validado formalmente.
- Los resultados deben ser revisados por personal educativo competente y junto con
  otras fuentes de información.

## Resolución de problemas

### `XGBoostError: input stream corrupted`

El archivo antiguo `modelos.joblib` contiene objetos internos de XGBoost incompatibles
entre entornos. Vuelve a exportar los modelos y utiliza los cinco archivos `.ubj`,
`modelos_sklearn.joblib` y `modelos_manifest.json`.

### No se encuentran los archivos de modelos

Comprueba que los ocho archivos generados están directamente dentro de `artefactos` y
que estás ejecutando el `app.py` actualizado.

### Las modificaciones no aparecen

Comprueba la ruta del archivo ejecutado:

```powershell
python -c "from pathlib import Path; print(Path('app.py').resolve())"
```

Después detén Streamlit, limpia la caché y vuelve a iniciarlo.

### Streamlit Cloud no encuentra el repositorio

Verifica que la rama sea `main`, que el archivo principal sea `app.py` y que Streamlit
tenga autorización para acceder al repositorio privado.
