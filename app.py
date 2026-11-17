import json
import numpy as np
import streamlit as st
from google import genai
from google.genai import types

st.set_page_config(page_title="Asistente · Ferretería López", page_icon="🔧")
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])   # la key va en Secrets
MAX_MENSAJES = 20          # por sesión: protege tu cuota y tu presupuesto
MOSTRAR_SIMILITUD = True   # True mientras calibras el UMBRAL

# ---------- Núcleo del chatbot (igual en Streamlit y en Gradio) ----------
MODELO = "gemini-3.1-flash-lite"
MODELO_EMB = "gemini-embedding-2"
DIM = 768
UMBRAL = 0.50          # CALÍBRALO con tus casos (ver la guía): no copies este valor
VENTANA = 6            # mensajes del historial que se envían al modelo
CONTACTO = "WhatsApp 442 000 0000, lunes a sábado de 8 a 19 h"

SISTEMA = f"""Eres el asistente automático de atención a clientes de Ferretería López,
una ferretería con 3 sucursales en Querétaro. Atiendes por chat en el sitio web.

# Alcance
Respondes sobre productos, pedidos, devoluciones, garantías, facturación, pagos,
envíos y sucursales, usando SOLO la información de <documentos>. Si la respuesta no
está ahí, dilo y ofrece el contacto humano; no completes con conocimiento general.
Si te preguntan algo que no tiene que ver con la ferretería, responde amablemente que
solo puedes ayudar con temas de Ferretería López.

# Estilo
Español de México, de usted, cálido y directo. Máximo 4 líneas, en texto corrido,
porque el chat se lee en el celular.

# Límites
- No inventes precios, existencias ni fechas de entrega: el cliente los toma como un
  compromiso de la empresa.
- No hagas excepciones a las políticas; las autoriza un supervisor.
- No pidas datos de pago ni contraseñas.

# Derivación a una persona
Si el cliente está molesto, presenta una queja, pide una excepción, menciona un cobro
indebido o pide hablar con alguien: reconoce la situación en una frase y dale este
contacto: {CONTACTO}."""

def _embed(contenido):
    return client.models.embed_content(model=MODELO_EMB, contents=contenido,
        config=types.EmbedContentConfig(output_dimensionality=DIM)).embeddings

def crear_indice(docs, lote=20):
    vectores = []
    for i in range(0, len(docs), lote):
        partes = [types.Content(parts=[types.Part.from_text(
                    text=f"title: {d['titulo']} | text: {d['texto']}")]) for d in docs[i:i + lote]]
        vectores += [e.values for e in _embed(partes)]
    M = np.array(vectores)
    return M / np.linalg.norm(M, axis=1, keepdims=True)

def buscar(consulta, docs, M, k=3):
    q = np.array(_embed(f"task: search result | query: {consulta}")[0].values)
    sims = M @ (q / np.linalg.norm(q))
    orden = np.argsort(-sims)[:k]
    return [(docs[i], float(sims[i])) for i in orden]

def responder(pregunta, historial, docs, M):
    """historial: lista de {"role": "user"|"assistant", "content": str}"""
    # La pregunta de seguimiento ("¿y si llegó roto?") se busca junto con el mensaje anterior
    previo = [m["content"] for m in historial if m["role"] == "user"][-1:]
    fragmentos = buscar(" ".join(previo + [pregunta]), docs, M)
    mejor = fragmentos[0][1]
    if mejor < UMBRAL:   # nada relevante: no se llama al modelo
        return (f"No tengo esa información. Con gusto le atiende una persona: {CONTACTO}.",
                [], mejor)
    contexto = "\n".join(f'<documento fuente="{d["fuente"]}">\n{d["texto"]}\n</documento>'
                         for d, _ in fragmentos)
    charla = "\n".join(("Cliente: " if m["role"] == "user" else "Asistente: ") + m["content"]
                       for m in historial[-VENTANA:])
    entrada = (f"<documentos>\n{contexto}\n</documentos>\n\n"
               f"<conversacion>\n{charla}\n</conversacion>\n\nCliente: {pregunta}")
    r = client.interactions.create(model=MODELO, system_instruction=SISTEMA, input=entrada)
    return r.output_text, [d for d, _ in fragmentos], mejor

# ---------- Interfaz ----------
with open("conocimiento.json", encoding="utf-8") as f:
    DOCS = json.load(f)

@st.cache_resource      # el índice se calcula una vez, no en cada mensaje
def indice():
    return crear_indice(DOCS)

st.title("🔧 Asistente de Ferretería López")
st.caption("Soy un asistente automático con inteligencia artificial y puedo equivocarme: "
           "confirme precios y existencias con una persona. No escriba datos de pago ni "
           "contraseñas. [Aviso de privacidad](https://ferreterialopez.mx/privacidad) · "
           f"¿Prefiere hablar con una persona? {CONTACTO}")

if "mensajes" not in st.session_state:
    st.session_state.mensajes = []
for m in st.session_state.mensajes:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if pregunta := st.chat_input("Escriba su pregunta"):
    if len(st.session_state.mensajes) >= 2 * MAX_MENSAJES:
        st.warning(f"Llegó al límite de mensajes de esta conversación. {CONTACTO}")
        st.stop()
    with st.chat_message("user"):
        st.markdown(pregunta)
    with st.chat_message("assistant"):
        try:
            texto, fuentes, sim = responder(pregunta, st.session_state.mensajes, DOCS, indice())
        except Exception:
            texto, fuentes, sim = (f"Tuve un problema técnico. Por favor contáctenos: {CONTACTO}.",
                                   [], 0.0)
        st.markdown(texto)
        if fuentes:
            with st.expander("Fuentes"):
                for d in fuentes:
                    st.markdown(f"- {d['titulo']} ({d['fuente']})")
        if MOSTRAR_SIMILITUD:
            st.caption(f"Similitud del mejor fragmento: {sim:.3f}")
    st.session_state.mensajes += [{"role": "user", "content": pregunta},
                                  {"role": "assistant", "content": texto}]
