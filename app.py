from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parent
RUTA_MODELOS_LEGACY = BASE_DIR / "artefactos" / "modelos.joblib"
RUTA_MODELOS_SKLEARN = BASE_DIR / "artefactos" / "modelos_sklearn.joblib"
RUTA_MANIFIESTO = BASE_DIR / "artefactos" / "modelos_manifest.json"
RUTA_METADATA = BASE_DIR / "artefactos" / "metadata.joblib"

CLASES_ESPERADAS = [0, 1, 2]
NUMERO_MODELOS_ESPERADO = 10
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

# Textos mostrados al usuario. Las claves numéricas siguen siendo exactamente
# las que reciben los modelos; únicamente cambia su representación visual.
INFORMACION_VARIABLES_DEFECTO = {
    "famsize": {
        "etiqueta": "Tamaño familiar",
        "descripcion": "Número de personas que componen la unidad familiar.",
        "opciones_texto": {
            0: "0 — 3 miembros o menos",
            1: "1 — Más de 3 miembros",
        },
    },
    "Pstatus": {
        "etiqueta": "Convivencia de los padres",
        "descripcion": "Situación de convivencia de los padres del alumno.",
        "opciones_texto": {
            0: "0 — Viven juntos",
            1: "1 — Viven separados",
        },
    },
    "educacion_familiar": {
        "etiqueta": "Nivel educativo familiar",
        "descripcion": "Nivel educativo representativo del entorno familiar.",
        "opciones_texto": {
            0: "0 — Sin estudios",
            1: "1 — Educación básica",
            2: "2 — Educación media",
            3: "3 — Educación alta",
            4: "4 — Estudios superiores universitarios",
        },
    },
    "apoyo_familiar": {
        "etiqueta": "Apoyo familiar",
        "descripcion": "Nivel de apoyo educativo recibido en el entorno familiar.",
        "opciones_texto": {
            0: "0 — Muy bajo",
            1: "1 — Bajo",
            2: "2 — Medio",
            3: "3 — Alto",
        },
    },
    "studytime": {
        "etiqueta": "Tiempo de estudio semanal",
        "descripcion": "Horas aproximadas que el alumno dedica a estudiar cada semana.",
        "opciones_texto": {
            1: "1 — Menos de 2 horas",
            2: "2 — Entre 2 y 5 horas",
            3: "3 — Entre 5 y 10 horas",
            4: "4 — Más de 10 horas",
        },
    },
    "failures": {
        "etiqueta": "Suspensos anteriores",
        "descripcion": "Número de asignaturas o cursos suspendidos anteriormente.",
        "opciones_texto": {
            0: "0 — Ninguno",
            1: "1 — Uno",
            2: "2 — Dos",
            3: "3 — Tres o más",
        },
    },
    "schoolsup": {
        "etiqueta": "Apoyo educativo del centro",
        "descripcion": "Indica si recibe apoyo educativo adicional del centro.",
        "opciones_texto": {
            0: "0 — No recibe",
            1: "1 — Sí recibe",
        },
    },
    "activities": {
        "etiqueta": "Actividades extraescolares",
        "descripcion": "Indica si participa habitualmente en actividades extraescolares.",
        "opciones_texto": {
            0: "0 — No participa",
            1: "1 — Sí participa",
        },
    },
    "higher": {
        "etiqueta": "Intención de cursar estudios superiores",
        "descripcion": "Indica si desea continuar con estudios superiores.",
        "opciones_texto": {
            0: "0 — No",
            1: "1 — Sí",
        },
    },
    "famrel": {
        "etiqueta": "Calidad de las relaciones familiares",
        "descripcion": "Valoración de la relación del alumno con su familia.",
        "opciones_texto": {
            1: "1 — Muy mala",
            2: "2 — Mala",
            3: "3 — Normal",
            4: "4 — Buena",
            5: "5 — Excelente",
        },
    },
    "freetime": {
        "etiqueta": "Tiempo libre después de clase",
        "descripcion": "Cantidad de tiempo libre disponible después de las clases.",
        "opciones_texto": {
            1: "1 — Muy poco",
            2: "2 — Poco",
            3: "3 — Moderado",
            4: "4 — Bastante",
            5: "5 — Mucho",
        },
    },
    "goout": {
        "etiqueta": "Frecuencia de salidas con amigos",
        "descripcion": "Frecuencia con la que el alumno sale con sus amistades.",
        "opciones_texto": {
            1: "1 — Muy baja",
            2: "2 — Baja",
            3: "3 — Moderada",
            4: "4 — Alta",
            5: "5 — Muy alta",
        },
    },
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
    .modelo-titulo {
        min-height: 3.2rem;
        margin-bottom: -0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def cargar_recursos():
    if not RUTA_METADATA.exists():
        return None, None

    metadata = joblib.load(RUTA_METADATA)

    # Formato portable: las regresiones se cargan mediante joblib y los modelos
    # XGBoost mediante el formato UBJSON estable de la propia biblioteca.
    if RUTA_MANIFIESTO.exists() or RUTA_MODELOS_SKLEARN.exists():
        if not RUTA_MANIFIESTO.exists() or not RUTA_MODELOS_SKLEARN.exists():
            raise FileNotFoundError(
                "La exportación está incompleta: deben existir tanto "
                "modelos_manifest.json como modelos_sklearn.joblib."
            )

        modelos_sklearn = joblib.load(RUTA_MODELOS_SKLEARN)
        with RUTA_MANIFIESTO.open(encoding="utf-8") as archivo:
            manifiesto = json.load(archivo)

        modelos = {}
        for entrada in manifiesto:
            nombre = str(entrada["nombre"])
            tipo = entrada["tipo"]

            if tipo == "sklearn":
                clave = entrada.get("clave", nombre)
                modelos[nombre] = modelos_sklearn[clave]
            elif tipo == "xgboost":
                ruta_modelo = BASE_DIR / "artefactos" / entrada["archivo"]
                if not ruta_modelo.exists():
                    raise FileNotFoundError(
                        f"No se encuentra el modelo XGBoost: {ruta_modelo.name}"
                    )
                modelo = XGBClassifier()
                modelo.load_model(ruta_modelo)
                modelos[nombre] = modelo
            else:
                raise ValueError(
                    f"Tipo de modelo no reconocido en el manifiesto: {tipo}"
                )

        return modelos, metadata

    # Compatibilidad temporal con la exportación antigua.
    if RUTA_MODELOS_LEGACY.exists():
        modelos = joblib.load(RUTA_MODELOS_LEGACY)
        return modelos, metadata

    return None, None


def validar_recursos(modelos, metadata):
    if not isinstance(modelos, dict) or len(modelos) != NUMERO_MODELOS_ESPERADO:
        raise ValueError(
            "La exportación debe proporcionar exactamente "
            f"{NUMERO_MODELOS_ESPERADO} modelos."
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


def informacion_variable(variable, metadata):
    informacion = INFORMACION_VARIABLES_DEFECTO.get(variable, {}).copy()
    informacion.update(
        metadata.get("informacion_variables", {}).get(variable, {})
    )
    return informacion


def descripcion_variable(variable, metadata):
    informacion = informacion_variable(variable, metadata)
    if informacion.get("etiqueta"):
        return str(informacion["etiqueta"])

    descripciones = metadata.get("descripciones_variables", {})
    return str(descripciones.get(variable, variable))


def descripcion_valor(variable, valor, metadata):
    """Devuelve una representación comprensible del valor si existe en metadata."""
    informacion = informacion_variable(variable, metadata)
    opciones_texto = informacion.get("opciones_texto", {})

    if valor in opciones_texto:
        return str(opciones_texto[valor])
    if str(valor) in opciones_texto:
        return str(opciones_texto[str(valor)])

    if isinstance(valor, (float, np.floating)):
        return f"{float(valor):g}"
    return str(valor)


def valor_inicial(serie):
    sin_nulos = serie.dropna()
    if sin_nulos.empty:
        return 0
    moda = sin_nulos.mode()
    return moda.iloc[0] if not moda.empty else sin_nulos.iloc[0]


def crear_widget_variable(variable, serie, metadata):
    """Crea un control a partir de los valores observados en los datos de referencia."""
    informacion = informacion_variable(variable, metadata)
    etiqueta = descripcion_variable(variable, metadata)
    descripcion = informacion.get("descripcion")
    ayuda = descripcion or (
        f"Variable original: {variable}" if etiqueta != variable else None
    )
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
        valor = st.selectbox(
            etiqueta,
            opciones,
            index=indice,
            format_func=lambda opcion: descripcion_valor(
                variable,
                opcion,
                metadata,
            ),
            help=ayuda,
            key=f"entrada_{variable}",
        )
        if descripcion:
            st.caption(descripcion)
        return valor

    if pd.api.types.is_integer_dtype(serie.dtype):
        minimo = int(sin_nulos.min())
        maximo = int(sin_nulos.max())
        inicial = int(round(float(sin_nulos.median())))
        valor = st.number_input(
            etiqueta,
            min_value=minimo,
            max_value=maximo,
            value=inicial,
            step=1,
            help=ayuda,
            key=f"entrada_{variable}",
        )
        if descripcion:
            st.caption(descripcion)
        return valor

    if pd.api.types.is_numeric_dtype(serie.dtype):
        minimo = float(sin_nulos.min())
        maximo = float(sin_nulos.max())
        inicial = float(sin_nulos.median())
        amplitud = maximo - minimo
        paso = max(amplitud / 100.0, 0.01)
        valor = st.number_input(
            etiqueta,
            min_value=minimo,
            max_value=maximo,
            value=inicial,
            step=paso,
            help=ayuda,
            format="%.4f",
            key=f"entrada_{variable}",
        )
        if descripcion:
            st.caption(descripcion)
        return valor

    opciones = sorted(str(valor) for valor in valores_unicos)
    valor = st.selectbox(
        etiqueta,
        opciones,
        help=ayuda,
        key=f"entrada_{variable}",
    )
    if descripcion:
        st.caption(descripcion)
    return valor


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


def grafica_probabilidades(probabilidades, nombres_clases):
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
            hovertemplate="%{x}<br>Grado de pertenencia: %{y:.2%}<extra></extra>",
        )
    )
    figura.update_layout(
        height=340,
        margin=dict(l=20, r=15, t=25, b=25),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        yaxis=dict(
            title="Grado de pertenencia",
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
def crear_explicador(clave_modelo, _modelo, _X_referencia):
    # clave_modelo forma parte de la caché y evita reutilizar un explicador
    # perteneciente a otro modelo. Los parámetros con _ se excluyen del hash.
    del clave_modelo
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


def calcular_explicacion_local(
    nombre_modelo,
    modelo,
    X_referencia,
    registro,
    clase_predicha,
):
    X_modelo_ref = preparar_registro_para_modelo(modelo, X_referencia)
    X_modelo_registro = preparar_registro_para_modelo(modelo, registro)
    explicador, tipo = crear_explicador(nombre_modelo, modelo, X_modelo_ref)

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
            }
        )
    return explicaciones


def unir_enumeracion(elementos):
    if not elementos:
        return ""
    if len(elementos) == 1:
        return elementos[0]
    return ", ".join(elementos[:-1]) + " y " + elementos[-1]


def generar_explicacion_natural(
    explicaciones,
    clase_predicha,
    probabilidad,
    nombres_clases,
    metadata,
):
    """Genera una explicación local legible sin atribuir causalidad a SHAP."""
    favorables = []
    contrarias = []
    neutras = []

    for elemento in explicaciones:
        variable_original = elemento["variable"]
        variable = descripcion_variable(variable_original, metadata)
        valor = descripcion_valor(variable_original, elemento["valor"], metadata)
        texto_variable = f"**{variable}** (valor: {valor})"
        contribucion = elemento["shap"]

        if contribucion > 1e-9:
            favorables.append(texto_variable)
        elif contribucion < -1e-9:
            contrarias.append(texto_variable)
        else:
            neutras.append(texto_variable)

    partes = [
        f"El modelo asigna principalmente al alumno a **{nombres_clases[clase_predicha]}** "
        f"con un grado de pertenencia del **{probabilidad:.1%}**."
    ]

    if favorables:
        partes.append(
            f"{unir_enumeracion(favorables)} "
            + ("es la variable que más impulsa" if len(favorables) == 1 else "son las variables que más impulsan")
            + " la predicción hacia esta clase."
        )
    if contrarias:
        partes.append(
            f"En cambio, {unir_enumeracion(contrarias)} "
            + ("actúa" if len(contrarias) == 1 else "actúan")
            + " en sentido contrario y reduce su puntuación."
        )
    if neutras:
        partes.append(
            f"{unir_enumeracion(neutras)} presenta una contribución prácticamente neutra."
        )

    return " ".join(partes)


st.title("🎓 Estimación de riesgo académico")
st.caption(
    "Introduce los datos del alumno para comparar las probabilidades de diez modelos "
    "y consultar una explicación local de sus predicciones."
)

try:
    modelos, metadata = cargar_recursos()
except Exception as error:
    st.error(
        "No ha sido posible cargar los modelos. Comprueba que copiaste todos los "
        "archivos generados dentro de la carpeta artefactos."
    )
    st.code(str(error), language="text")
    st.stop()

if modelos is None or metadata is None:
    st.error(
        "No se encuentran los archivos exportados. Descomprime artefactos.zip y "
        "copia todos sus archivos dentro de la carpeta artefactos del proyecto."
    )
    st.code(
        "artefactos/modelos_sklearn.joblib\n"
        "artefactos/modelos_manifest.json\n"
        "artefactos/xgboost_01.ubj ... xgboost_05.ubj\n"
        "artefactos/metadata.joblib",
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
    st.info(
        "Los controles y sus límites se generan con los datos de referencia empleados "
        "al exportar los modelos."
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
columna_3.metric(
    "Coincidencia de modelos",
    f"{votos} de {len(resultados)}",
)

st.caption(
    "El resultado conjunto es el promedio simple de las probabilidades de los diez "
    "modelos. Debe interpretarse como apoyo a la decisión, no como diagnóstico ni como "
    "certeza garantizada."
)

st.subheader("Resultados de los diez modelos")
st.caption(
    "Cada gráfica muestra el grado de pertenencia estimado para las tres clases. "
    "Debajo se explican las tres variables con mayor influencia local según SHAP."
)

for inicio in range(0, len(resultados), 2):
    columnas_graficas = st.columns(2)
    for columna, resultado in zip(columnas_graficas, resultados[inicio:inicio + 2]):
        with columna:
            with st.container(border=True):
                st.markdown(
                    f'<div class="modelo-titulo"><h3>{resultado["nombre"]}</h3></div>',
                    unsafe_allow_html=True,
                )
                st.plotly_chart(
                    grafica_probabilidades(
                        resultado["probabilidades"],
                        nombres_clases,
                    ),
                    use_container_width=True,
                    key=f"grafica_{inicio}_{resultado['nombre']}",
                )

                clase = resultado["clase_predicha"]
                probabilidad = float(resultado["probabilidades"][clase])

                try:
                    explicaciones = calcular_explicacion_local(
                        resultado["nombre"],
                        resultado["modelo"],
                        X_referencia,
                        registro,
                        clase,
                    )
                    st.markdown(
                        generar_explicacion_natural(
                            explicaciones,
                            clase,
                            probabilidad,
                            nombres_clases,
                            metadata,
                        )
                    )
                    detalle_shap = " · ".join(
                        f"{descripcion_variable(item['variable'], metadata)}: "
                        f"{item['shap']:+.4f}"
                        for item in explicaciones
                    )
                    st.caption(f"Contribuciones SHAP: {detalle_shap}")
                except Exception as error:
                    st.warning(
                        "La predicción se calculó correctamente, pero SHAP no pudo "
                        f"generar la explicación local: {error}"
                    )

st.caption(
    "Las explicaciones SHAP describen el comportamiento de cada modelo para este caso. "
    "Indican asociación con la predicción, no una relación causal con el rendimiento."
)

with st.expander("Datos introducidos"):
    st.dataframe(registro, use_container_width=True, hide_index=True)

st.caption(
    "Uso responsable: los resultados deben revisarse por personal competente y no deben "
    "utilizarse como único criterio para adoptar decisiones que afecten al alumnado."
)
