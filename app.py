import streamlit as st
import pandas as pd
import sqlite3
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 0rem;
        }
    </style>
""", unsafe_allow_html=True)

# 1. Configuración de la página
st.set_page_config(page_title="BI Sales Chatbot", page_icon="📊", layout="centered")

st.title("🤖 Chatbot de Ventas BI")
st.markdown("Sube tu reporte de ventas y consúltalo en lenguaje natural. El modelo generará la consulta SQL y extraerá el dato exacto.")

# 2. Inicializar el historial de chat en la sesión
if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def init_chain():
    # Detectar si estamos en la nube
    # Usamos Llama 3 a través de Groq y leemos la API Key de los secretos de Streamlit
    try:
        from langchain_groq import ChatGroq
        api_key = st.secrets["GROQ_API_KEY"]
        llm = ChatGroq(api_key=api_key, model_name="openai/gpt-oss-20b", temperature=0)
    
    # Si da error (estamos en local/Docker sin el archivo secrets), usar Ollama
    except Exception:
        import os
        from langchain_ollama import ChatOllama
        ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        llm = ChatOllama(model="llama3.2", temperature=0, base_url=ollama_url)
    
    template = """Eres un Data Analyst experto en SQLite. 
    Tienes una tabla 'ventas' con dos columnas: 'Producto' (TEXT) y 'Unidades_Vendidas' (INTEGER).
    
    REGLA DE NEGOCIO (Taxonomía):
    - Calzado = 'Zapatos', 'Medias'
    - Ropa = 'Pantalones', 'Camisetas', 'Camperas'
    - Accesorios = 'Sombreros', 'Guantes'
    (Usa estas categorías en un WHERE Producto IN (...) SOLO si preguntan por "ropa", "calzado" o "accesorios").

    REGLAS DE SEGURIDAD (INTEGRIDAD DE DATOS):
    1. FUERA DE DOMINIO: Si el usuario hace una pregunta que NO tiene NADA que ver con las ventas o los productos de la tabla proporcionada (ej. clima, deportes, saludos, chistes), debes devolver EXACTAMENTE esta consulta:
       SELECT 'Lo siento, como Analista de BI solo puedo responder preguntas sobre el reporte de ventas.' AS Mensaje;
    2. CORTAFUEGOS SEMÁNTICO: Si el usuario menciona una categoría, atributo, temporada (ej. 'invierno', 'verano') o adjetivo que NO existe explícitamente en la tabla o en la regla de Taxonomía, DEBES devolver la consulta de seguridad:
       SELECT 'Lo siento, como Analista de BI solo puedo responder sobre las categorías oficiales (Ropa, Calzado, Accesorios).' AS Mensaje;

    REGLA DE SINTAXIS OBLIGATORIA:
    - TODA consulta que busque datos numéricos o evalúe columnas DEBE incluir obligatoriamente la cláusula "FROM ventas". Nunca omitas el origen de los datos.

    REGLAS ESTRICTAS DE SINTAXIS SQL:
    1. El orden OBLIGATORIO es: SELECT -> FROM -> WHERE -> ORDER BY -> LIMIT -> OFFSET.
    2. NUNCA pongas un WHERE después de un ORDER BY, LIMIT u OFFSET.

    EJEMPLOS DE CÓDIGO (Aprende de estos patrones):
    Usuario: ¿Cuál es el producto más vendido?
    SQL: SELECT Producto, Unidades_Vendidas FROM ventas ORDER BY Unidades_Vendidas DESC LIMIT 1;
    
    Usuario: ¿Cuál es el producto más vendido después de las Medias?
    SQL: SELECT Producto FROM ventas WHERE Unidades_Vendidas < (SELECT Unidades_Vendidas FROM ventas WHERE Producto = 'Medias') ORDER BY Unidades_Vendidas DESC LIMIT 1;
    
    Usuario: ¿Cuál es el segundo producto más vendido?
    SQL: SELECT Producto FROM ventas WHERE Unidades_Vendidas < (SELECT MAX(Unidades_Vendidas) FROM ventas) ORDER BY Unidades_Vendidas DESC LIMIT 1;
    
    Usuario: ¿Cuántos guantes se vendieron?
    SQL: SELECT Unidades_Vendidas FROM ventas WHERE Producto = 'Guantes';
    
    Usuario: ¿Qué es lo que menos salió?
    SQL: SELECT Producto, Unidades_Vendidas FROM ventas ORDER BY Unidades_Vendidas ASC LIMIT 1;

    Usuario: ¿Qué productos vendieron más de 60 unidades?
    SQL: SELECT Producto FROM ventas WHERE Unidades_Vendidas > 60;
    
    Usuario: ¿Cómo está el clima?
    SQL: SELECT 'Lo siento, como Analista de BI solo puedo responder preguntas sobre el reporte de ventas.' AS Mensaje;

    Usuario: Dame el total de ventas de indumentaria de invierno.
    SQL: SELECT 'Lo siento, como Analista de BI solo puedo responder sobre las categorías oficiales (Ropa, Calzado, Accesorios).' AS Mensaje;

    Usuario: ¿Cuál es el total de ventas de remeras y guantes?
    SQL: SELECT SUM(Unidades_Vendidas) FROM ventas WHERE Producto IN ('Camisetas', 'Guantes');

    Usuario: ¿Cuál es el stock de calzado?
    SQL: SELECT 'Lo siento, el reporte solo contiene métricas de ventas, no manejamos información de stock.' AS Mensaje;
    
    PREGUNTA DEL USUARIO: {pregunta}
    
    Escribe la consulta SQL exacta para responder. 
    IMPORTANTE: Devuelve ÚNICAMENTE el código SQL. No uses bloques de código (```sql), no des explicaciones.
    """
    prompt = PromptTemplate.from_template(template)
    return prompt | llm | StrOutputParser()

sql_chain = init_chain()

# 4. Widget para subir el archivo CSV
uploaded_file = st.file_uploader("Sube tu archivo de ventas (.csv)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    conn = sqlite3.connect(':memory:')
    df.to_sql('ventas', conn, index=False, if_exists='replace')
    
    with st.expander("Ver vista previa de los datos"):
        st.dataframe(df)

    # 5. Mostrar historial de chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 6. Input del usuario
    if prompt := st.chat_input("Ej: ¿Cuál es el segundo producto más vendido?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Traduciendo a SQL y analizando datos..."):
                try:
                    query_generada = sql_chain.invoke({"pregunta": prompt}).strip()
                    query_generada = query_generada.replace("```sql", "").replace("```", "").strip()
                    
                    st.code(query_generada, language="sql")
                    
                    cursor = conn.cursor()
                    cursor.execute(query_generada)
                    resultados = cursor.fetchall()
                    
                    if not resultados or resultados[0][0] is None:
                        respuesta = "No se encontraron resultados para esa consulta."
                    else:
                        respuesta = "\n".join([f"- {fila[0]}: {fila[1]}" if len(fila) > 1 else f"- {fila[0]}" for fila in resultados])
                    
                    st.markdown(f"**Resultado:**\n{respuesta}")
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": f"**Query ejecutada:**\n```sql\n{query_generada}\n```\n**Resultado:**\n{respuesta}"
                    })

                except Exception as e:
                    error_msg = f"⚠️ Hubo un error de ejecución SQL: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
else:
    st.info("👆 Por favor, sube un archivo CSV para habilitar el chat.")
