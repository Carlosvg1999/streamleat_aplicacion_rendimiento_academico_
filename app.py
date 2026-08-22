from pathlib import Path
import html

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
RUTA_MODELOS = BASE_DIR / "artefactos" / "modelos.joblib"
RUTA_METADATA = BASE_DIR / "artefactos" / "metadata.joblib"

CLASES_ESPERADAS = [0, 1, 2]
NOMBRES_CLASES_DEFECTO = {
    0: "Mucho riesgo",
    1: "Riesgo",
    2: "Bajo riesgo",
}
COLORES_CLASES = {
    0: "#DC4C64",
    1: "#F2A900",
    2: "#2E8B57",
}


st.set_page_config(
    page_title="Estimación de riesgo académico",
    page_icon="🎓",
    layout="wide",
)


st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem;}
    [data-testid="stMetric"] {
        background: #f7f8fa;
        border: 1px solid #e2e5e9;
        border-radius: 12px;
        padding: 14px 16px;
    }
    .explicacion {
        border-left: 4px solid #5b6cff;
        background: #f7f8ff;
        border-radius: 6px;
        padding: 10px 14px;
        margin: 8px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def cargar_recursos():
    if not RUTA_MODELOS.exists() or not RUTA_METADATA.exists():
        return None, None

    modelos = joblib.load(RUTA_MODELOS)
    metadata = joblib.load(RUTA_METADATA)
    return modelos, metadata


def validar_recursos(modelos, metadata):
    if not isinstance(modelos, dict) or len(modelos) != 4:
        raise ValueError(
            "modelos.joblib debe contener un diccionario con exactamente cuatro modelos."
        )

    modelos_invalidos = [
        nombre
        for nombre, modelo in modelos.items()
        if not hasattr(modelo, "predict_proba")
    ]
    if modelos_invalidos:
        raise ValueError(
            "Estos modelos no disponen de predict_proba: "
            + ", ".join(modelos_invalidos)
        )

    if not isinstance(metadata, dict) or "X_referencia" not in metadata:
        raise ValueError("metadata.joblib debe incluir la clave 'X_referencia'.")

    X_referencia = metadata["X_referencia"]
    if not isinstance(X_referencia, pd.DataFrame) or X_referencia.empty:
        raise ValueError("X_referencia debe ser un DataFrame no vacío.")

    if X_referencia.columns.duplicated().any():
        raise ValueError("X_referencia contiene nombres de columnas duplicados.")

    for nombre, modelo in modelos.items():
        clases = {int(clase) for clase in np.asarray(modelo.classes_)}
        if clases != set(CLASES_ESPERADAS):
            raise ValueError(
                f"{nombre} utiliza las clases {sorted(clases)}; se esperaban [0, 1, 2]."
            )

    return X_referencia


def nombres_clases_desde_metadata(metadata):
    nombres = metadata.get("nombres_clases", NOMBRES_CLASES_DEFECTO)
    return {
        clase: str(nombres.get(clase, nombres.get(str(clase), NOMBRES_CLASES_DEFECTO[clase])))
        for clase in CLASES_ESPERADAS
    }


def descripcion_variable(variable, metadata):
    descripciones = metadata.get("descripciones_variables", {})
    return str(descripciones.get(variable, variable))


def valor_inicial(serie):
    sin_nulos = serie.dropna()
    if sin_nulos.empty:
        return 0
    moda = sin_nulos.mode()
    return moda.iloc[0] if not moda.empty else sin_nulos.iloc[0]


def crear_widget_variable(variable, serie, metadata):
    """Crea un control a partir de los valores observados en los datos de referencia."""
    etiqueta = descripcion_variable(variable, metadata)
    ayuda = f"Variable original: {variable}" if etiqueta != variable else None
    sin_nulos = serie.dropna()
    valores_unicos = list(pd.unique(sin_nulos))

    # Las variables binarias, ordinales o categóricas pequeñas se muestran como selector.
    if len(valores_unicos) <= 12:
        try:
            opciones = sorted(valores_unicos)
        except TypeError:
            opciones = valores_unicos

        inicial = valor_inicial(serie)
        indice = opciones.index(inicial) if inicial in opciones else 0
        return st.selectbox(etiqueta, opciones, index=indice, help=ayuda)

    if pd.api.types.is_integer_dtype(serie.dtype):
        minimo = int(sin_nulos.min())
        maximo = int(sin_nulos.max())
        inicial = int(round(float(sin_nulos.median())))
        return st.number_input(
            etiqueta,
            min_value=minimo,
            max_value=maximo,
            value=inicial,
            step=1,
            help=ayuda,
        )

    if pd.api.types.is_numeric_dtype(serie.dtype):
        minimo = float(sin_nulos.min())
        maximo = float(sin_nulos.max())
        inicial = float(sin_nulos.median())
        amplitud = maximo - minimo
        paso = max(amplitud / 100.0, 0.01)
        return st.number_input(
            etiqueta,
            min_value=minimo,
            max_value=maximo,
            value=inicial,
            step=paso,
            help=ayuda,
            format="%.4f",
        )

    opciones = sorted(str(valor) for valor in valores_unicos)
    return st.selectbox(etiqueta, opciones, help=ayuda)


def construir_registro(X_referencia, metadata):
    valores = {}
    columnas = list(X_referencia.columns)
    numero_columnas = min(3, max(1, len(columnas)))
    columnas_ui = st.columns(numero_columnas)

    for posicion, variable in enumerate(columnas):
        with columnas_ui[posicion % numero_columnas]:
            valores[variable] = crear_widget_variable(
                variable,
                X_referencia[variable],
                metadata,
            )

    registro = pd.DataFrame([valores], columns=columnas)

    # Recupera los tipos originales siempre que sea posible.
    for variable in columnas:
        try:
            registro[variable] = registro[variable].astype(
                X_referencia[variable].dtype
            )
        except (TypeError, ValueError):
            pass

    return registro


def preparar_registro_para_modelo(modelo, registro):
    if hasattr(modelo, "feature_names_in_"):
        columnas = list(modelo.feature_names_in_)
        faltantes = [columna for columna in columnas if columna not in registro.columns]
        if faltantes:
            raise ValueError(f"Faltan variables para el modelo: {faltantes}")
        return registro[columnas]
    return registro


def ordenar_probabilidades(modelo, probabilidades):
    salida = np.zeros(3, dtype=float)
    for posicion, clase in enumerate(np.asarray(modelo.classes_)):
        clase = int(clase)
        if clase in CLASES_ESPERADAS:
            salida[clase] = float(probabilidades[posicion])
    return salida


def grafica_probabilidades(nombre_modelo, probabilidades, nombres_clases):
    etiquetas = [
        f"Clase {clase}<br>{nombres_clases[clase]}" for clase in CLASES_ESPERADAS
    ]
    figura = go.Figure(
        go.Bar(
            x=etiquetas,
            y=probabilidades,
            marker_color=[COLORES_CLASES[c] for c in CLASES_ESPERADAS],
            text=[f"{valor:.1%}" for valor in probabilidades],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{x}<br>Probabilidad estimada: %{y:.2%}<extra></extra>",
        )
    )
    figura.update_layout(
        title=nombre_modelo,
        height=390,
        margin=dict(l=30, r=20, t=65, b=35),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        yaxis=dict(
            title="Probabilidad estimada",
            tickformat=".0%",
            range=[0, 1.10],
            gridcolor="#e7e9ed",
        ),
    )
    return figura


def normalizar_valores_shap(explicacion, numero_variables):
    if isinstance(explicacion, list):
        valores = np.stack(
            [np.asarray(elemento.values if hasattr(elemento, "values") else elemento)
             for elemento in explicacion],
            axis=-1,
        )
    elif hasattr(explicacion, "values"):
        valores = np.asarray(explicacion.values)
    else:
        valores = np.asarray(explicacion)

    if valores.ndim == 1:
        valores = valores[np.newaxis, :, np.newaxis]
    elif valores.ndim == 2:
        # Una observación multiclase puede llegar como variables x clases.
        if valores.shape[0] == numero_variables and valores.shape[1] != numero_variables:
            valores = valores[np.newaxis, :, :]
        else:
            valores = valores[:, :, np.newaxis]

    if (
        valores.ndim == 3
        and valores.shape[1] != numero_variables
        and valores.shape[2] == numero_variables
    ):
        valores = np.transpose(valores, (0, 2, 1))

    if valores.ndim != 3 or valores.shape[1] != numero_variables:
        raise ValueError(f"Formato SHAP no reconocido: {valores.shape}")

    return valores


@st.cache_resource(show_spinner=False)
def crear_explicador(_modelo, _X_referencia):
    nombre_clase = _modelo.__class__.__name__.lower()
    referencia = shap.sample(
        _X_referencia,
        min(100, len(_X_referencia)),
        random_state=42,
    )

    if "xgb" in nombre_clase or "forest" in nombre_clase or "tree" in nombre_clase:
        try:
            return shap.TreeExplainer(_modelo), "directo"
        except Exception:
            pass

    if "logisticregression" in nombre_clase:
        try:
            return shap.LinearExplainer(_modelo, referencia), "directo"
        except Exception:
            pass

    # Respaldo independiente del tipo de estimador; puede ser más lento.
    return shap.Explainer(_modelo.predict_proba, referencia), "generico"


def calcular_explicacion_local(modelo, X_referencia, registro, clase_predicha):
    X_modelo_ref = preparar_registro_para_modelo(modelo, X_referencia)
    X_modelo_registro = preparar_registro_para_modelo(modelo, registro)
    explicador, tipo = crear_explicador(modelo, X_modelo_ref)

    if tipo == "generico":
        explicacion = explicador(
            X_modelo_registro,
            max_evals=max(2 * X_modelo_registro.shape[1] + 1, 11),
        )
    else:
        explicacion = explicador(X_modelo_registro)

    valores = normalizar_valores_shap(
        explicacion,
        numero_variables=X_modelo_registro.shape[1],
    )[0]

    posiciones = np.where(np.asarray(modelo.classes_).astype(int) == clase_predicha)[0]
    posicion_clase = int(posiciones[0])

    if valores.shape[1] == 1:
        contribuciones = valores[:, 0]
    else:
        contribuciones = valores[:, posicion_clase]

    indices = np.argsort(np.abs(contribuciones))[::-1][:3]
    explicaciones = []
    for indice in indices:
        contribucion = float(contribuciones[indice])
        variable = X_modelo_registro.columns[indice]
        valor = X_modelo_registro.iloc[0, indice]
        explicaciones.append(
            {
                "variable": variable,
                "valor": valor,
                "shap": contribucion,
                "direccion": "favorece" if contribucion > 0 else "reduce",
            }
        )
    return explicaciones


def mostrar_explicaciones(explicaciones, clase_predicha, nombres_clases, metadata):
    for posicion, elemento in enumerate(explicaciones, start=1):
        variable = descripcion_variable(elemento["variable"], metadata)
        valor = elemento["valor"]
        contribucion = elemento["shap"]
        if contribucion > 0:
            icono = "↑"
            texto = f"favorece la estimación de {nombres_clases[clase_predicha]}"
            color = "#237a4b"
        elif contribucion < 0:
            icono = "↓"
            texto = f"reduce la estimación de {nombres_clases[clase_predicha]}"
            color = "#b33b4b"
        else:
            icono = "→"
            texto = "tiene un efecto prácticamente neutro"
            color = "#666666"

        st.markdown(
            f"""
            <div class="explicacion">
              <b>{posicion}. {html.escape(str(variable))}</b> = {html.escape(str(valor))}<br>
              <span style="color:{color}">{icono} {texto}</span>
              <span style="color:#6b7280"> · SHAP {contribucion:+.4f}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


st.title("Estimación de riesgo académico")
st.caption(
    "Introduce los datos del alumno para comparar las probabilidades de cuatro modelos "
    "y consultar una explicación local de sus predicciones."
)

modelos, metadata = cargar_recursos()

if modelos is None or metadata is None:
    st.error(
        "No se encuentran los archivos de modelos. Ejecuta primero el bloque de "
        "exportación incluido en README.md y coloca los dos .joblib en la carpeta "
        "artefactos."
    )
    st.code(
        "artefactos/modelos.joblib\nartefactos/metadata.joblib",
        language="text",
    )
    st.stop()

try:
    X_referencia = validar_recursos(modelos, metadata)
except Exception as error:
    st.error(f"Los artefactos no son válidos: {error}")
    st.stop()

nombres_clases = nombres_clases_desde_metadata(metadata)

with st.form("formulario_alumno"):
    st.subheader("Datos del alumno")
    with st.expander(
        "Descripción de las variables y sus valores",
        expanded=True
    ):

        st.markdown("""
    | Variable | Descripción | Significado de los valores |
    |---|---|---|
    | **famsize** | Tamaño de la familia del alumno. | `0`: más de 3 miembros · `1`: 3 miembros o menos |
    | **Pstatus** | Situación de convivencia de los padres. | `0`: separados · `1`: viven juntos |
    | **educacion_familiar** | Nivel educativo familiar. | `0`: sin estudios · `1`: educación primaria · `2`: educación básica · `3`: educación secundaria · `4`: estudios superiores |
    | **apoyo_familiar** | Nivel de apoyo educativo recibido por parte de la familia. | `0`: bajo · `1`: medio · `2`: alto |
    | **failures** | Número de suspensos anteriores del alumno. | `0`: ninguno · `1`: uno · `2`: dos · `3`: tres o más |
    | **schoolsup** | Apoyo educativo adicional proporcionado por el centro. | `0`: no recibe · `1`: sí recibe |
    | **activities** | Participación en actividades extraescolares. | `0`: no participa · `1`: sí participa |
    | **higher** | Intención de cursar estudios superiores. | `0`: no · `1`: sí |
    | **famrel** | Calidad de las relaciones familiares. | Escala de `1` — muy malas a `5` — excelentes |
    | **freetime** | Tiempo libre disponible después de las clases. | Escala de `1` — muy poco a `5` — mucho |
    | **goout** | Frecuencia con la que el alumno sale con amigos. | Escala de `1` — muy baja a `5` — muy alta |
        """)

    st.caption(
        "Los valores deben introducirse conforme a estas codificaciones, "
        "que son las utilizadas durante el entrenamiento de los modelos."
    )
    registro = construir_registro(X_referencia, metadata)
    enviado = st.form_submit_button(
        "Estimar nivel de riesgo",
        type="primary",
        use_container_width=True,
    )

if not enviado:
    st.stop()

resultados = []
errores = []
for nombre_modelo, modelo in modelos.items():
    try:
        registro_modelo = preparar_registro_para_modelo(modelo, registro)
        probabilidades_crudas = modelo.predict_proba(registro_modelo)[0]
        probabilidades = ordenar_probabilidades(modelo, probabilidades_crudas)
        clase_predicha = int(CLASES_ESPERADAS[int(np.argmax(probabilidades))])
        resultados.append(
            {
                "nombre": nombre_modelo,
                "modelo": modelo,
                "probabilidades": probabilidades,
                "clase_predicha": clase_predicha,
            }
        )
    except Exception as error:
        errores.append(f"{nombre_modelo}: {error}")

if errores:
    st.error("No se pudieron calcular todas las predicciones:\n\n" + "\n\n".join(errores))
    st.stop()

probabilidad_media = np.mean(
    [resultado["probabilidades"] for resultado in resultados],
    axis=0,
)
clase_conjunta = int(np.argmax(probabilidad_media))
confianza_conjunta = float(probabilidad_media[clase_conjunta])
votos = sum(
    resultado["clase_predicha"] == clase_conjunta for resultado in resultados
)

st.divider()
st.subheader("Resultado estimado")
columna_1, columna_2, columna_3 = st.columns(3)
columna_1.metric(
    "Nivel de riesgo conjunto",
    f"Clase {clase_conjunta} · {nombres_clases[clase_conjunta]}",
)
columna_2.metric("Probabilidad media", f"{confianza_conjunta:.1%}")
columna_3.metric("Coincidencia de modelos", f"{votos} de 4")

st.caption(
    "El resultado conjunto es el promedio simple de las probabilidades de los cuatro "
    "modelos. Debe interpretarse como apoyo a la decisión, no como diagnóstico ni como "
    "certeza garantizada."
)

st.subheader("Probabilidades de los cuatro modelos")
for inicio in range(0, 4, 2):
    columnas_graficas = st.columns(2)
    for columna, resultado in zip(columnas_graficas, resultados[inicio:inicio + 2]):
        with columna:
            st.plotly_chart(
                grafica_probabilidades(
                    resultado["nombre"],
                    resultado["probabilidades"],
                    nombres_clases,
                ),
                use_container_width=True,
                key=f"grafica_{inicio}_{resultado['nombre']}",
            )
            clase = resultado["clase_predicha"]
            st.write(
                f"Predicción: **clase {clase} · {nombres_clases[clase]}** "
                f"({resultado['probabilidades'][clase]:.1%})"
            )

st.subheader("Variables relevantes para cada predicción")
st.caption(
    "Se muestran las tres contribuciones SHAP de mayor magnitud para la clase predicha "
    "por cada modelo. La dirección explica el comportamiento del modelo, no una relación causal."
)

for resultado in resultados:
    with st.expander(resultado["nombre"], expanded=True):
        clase = resultado["clase_predicha"]
        st.write(
            f"Explicación de la predicción **clase {clase} · {nombres_clases[clase]}**"
        )
        try:
            explicaciones = calcular_explicacion_local(
                resultado["modelo"],
                X_referencia,
                registro,
                clase,
            )
            mostrar_explicaciones(
                explicaciones,
                clase,
                nombres_clases,
                metadata,
            )
        except Exception as error:
            st.warning(
                "La predicción se calculó correctamente, pero SHAP no pudo generar "
                f"la explicación local: {error}"
            )

with st.expander("Datos introducidos"):
    st.dataframe(registro, use_container_width=True, hide_index=True)

st.caption(
    "Uso responsable: los resultados deben revisarse por personal competente y no deben "
    "utilizarse como único criterio para adoptar decisiones que afecten al alumnado."
)
