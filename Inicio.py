# app.py — Demo TF-IDF en Español (mejorada)
import re
import unicodedata
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Opcional: stemming español (con fallback "identidad" si falta NLTK)
try:
    from nltk.stem import SnowballStemmer
    STEMMER = SnowballStemmer("spanish")
    def stem_es(t: str) -> str: return STEMMER.stem(t)
except Exception:
    def stem_es(t: str) -> str: return t  # sin stemming si NLTK no está

st.set_page_config(page_title="Demo TF-IDF en Español", page_icon="🔍", layout="wide")
st.title("🔍 Demo TF-IDF en Español – Laboratorio")

# -----------------------------------------------------------------------------
# Datos de ejemplo
# -----------------------------------------------------------------------------
DEFAULT_DOCS = """El perro ladra fuerte en el parque.
El gato maúlla suavemente durante la noche.
El perro y el gato juegan juntos en el jardín.
Los niños corren y se divierten en el parque.
La música suena muy alta en la fiesta.
Los pájaros cantan hermosas melodías al amanecer."""

SUGERIDAS = [
    "¿Dónde juegan el perro y el gato?",
    "¿Qué hacen los niños en el parque?",
    "¿Cuándo cantan los pájaros?",
    "¿Dónde suena la música alta?",
    "¿Qué animal maúlla durante la noche?",
]

# Stopwords ES compactas (sin depender de NLTK)
STOPWORDS_ES = set("""
a al algo algunas algunos ante antes así aún cada como con contra cuándo cuál cuáles cuando de del desde donde dos el ella ellas ellos en entre era erais eran eras eres es esa esas ese eso esos esta estaban estado estaba estáis estamos están estar estará estas este éstos esto estos estoy fue fuera fueron fui fuimos ha habían había habíais habían habíamos haber hace hacen hacer hacia han hasta hay la las le les lo los más me mi mis mucha muchas mucho muchos muy nada ni no nos nosotras nosotros o os otra otras otro otros para pero poco por porque qué que quien quién quienes se ser será si sido sin sobre sois somos son soy su sus también te tiene tienen tener tengo tenía tenían tenemos tenéis tus tu un una uno unas unos ya
""".split())

# -----------------------------------------------------------------------------
# Utils de limpieza/tokenización
# -----------------------------------------------------------------------------
def strip_accents(s: str) -> str:
    # Normaliza acentos (camión -> camion)
    nkfd = unicodedata.normalize("NFKD", s)
    return "".join([c for c in nkfd if not unicodedata.combining(c)])

def tokenize_es(text: str, use_stemming: bool, remove_stopwords: bool) -> List[str]:
    # Minúsculas + quitar URLs/menções
    text = text.lower()
    text = re.sub(r"https?://\S+|[@#]\w+", " ", text)
    # Mantener letras (incluye ñ, áéíóúü) y espacios; quitar números/otros signos
    text = re.sub(r"[^a-záéíóúüñ\s]", " ", text)
    # Normalizar espacios
    text = re.sub(r"\s+", " ", text).strip()
    # Opcional: quitar acentos para emparejar mejor
    text = strip_accents(text)
    tokens = [t for t in text.split() if len(t) > 1]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS_ES]
    if use_stemming:
        tokens = [stem_es(t) for t in tokens]
    return tokens

def build_vectorizer(ngram_max: int, min_df: float, max_df: float,
                     use_stemming: bool, remove_stopwords: bool,
                     sublinear: bool, norm: str) -> TfidfVectorizer:
    return TfidfVectorizer(
        tokenizer=lambda s: tokenize_es(s, use_stemming=use_stemming, remove_stopwords=remove_stopwords),
        ngram_range=(1, ngram_max),
        min_df=min_df,            # puede ser int o proporción
        max_df=max_df,            # idem
        use_idf=True,
        sublinear_tf=sublinear,
        norm=norm,                # "l2" recomendado
    )

def highlight_terms(text: str, terms: List[str]) -> str:
    """Resalta términos (ya normalizados sin acentos) dentro de text (tal cual).
       Hacemos búsqueda case-insensitive y de palabra completa.
    """
    if not terms: return text
    # Construir patrón con OR escapado
    safe = [re.escape(t) for t in sorted(set(terms), key=len, reverse=True)]
    if not safe: return text
    pattern = r"\b(" + "|".join(safe) + r")\b"
    def repl(m):
        return f"<mark>{m.group(0)}</mark>"
    # case-insensitive y sin acentos: aplicamos sobre versión sin acentos para localizar posiciones,
    # luego sustituimos sobre original con heurística sencilla
    # (estrategia simple: usamos regex sobre original pero con IGNORECASE)
    return re.sub(pattern, repl, text, flags=re.IGNORECASE)

# -----------------------------------------------------------------------------
# Sidebar — Parámetros
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Parámetros del vectorizador")
    ngram_max = st.select_slider("N-gramas (máx)", options=[1, 2, 3], value=2)
    min_df = st.number_input("min_df", value=1.0, min_value=0.0, step=0.1,
                             help="Si ≥1 se interpreta como conteo; si <1 como proporción del corpus.")
    max_df = st.number_input("max_df", value=1.0, min_value=0.0, step=0.1,
                             help="Filtra términos muy frecuentes (ruido). 1.0 = sin filtro")
    remove_stop = st.checkbox("Quitar stopwords", True)
    use_stem = st.checkbox("Aplicar stemming (Snowball)", True)
    sublinear = st.checkbox("Sublinear TF (log(1+tf))", True)
    norm = st.selectbox("Normalización", ["l2", None], index=0)

    st.markdown("---")
    topk_matrix = st.slider("Top-K términos para mostrar en matriz", 5, 50, 20, 1)
    show_all_tokens = st.checkbox("Mostrar TODAS las columnas (cuidado: puede ser grande)", False)

# -----------------------------------------------------------------------------
# Layout principal
# -----------------------------------------------------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    text_input = st.text_area("📝 Documentos (uno por línea):", DEFAULT_DOCS, height=180)
    question = st.text_input("❓ Escribe tu pregunta:", "¿Dónde juegan el perro y el gato?")

with col2:
    st.markdown("### 💡 Preguntas sugeridas")
    for q in SUGERIDAS:
        if st.button(q, use_container_width=True):
            st.session_state.question = q
            st.rerun()

# Sincroniza si se escogió sugerida
if "question" in st.session_state:
    question = st.session_state.question

# -----------------------------------------------------------------------------
# Botón Analizar
# -----------------------------------------------------------------------------
if st.button("🔎 Analizar", type="primary"):
    docs = [d.strip() for d in text_input.split("\n") if d.strip()]
    if len(docs) < 1:
        st.error("⚠️ Ingresa al menos un documento.")
        st.stop()
    if not question.strip():
        st.error("⚠️ Escribe una pregunta.")
        st.stop()

    # Construir vectorizador con parámetros
    vec = build_vectorizer(
        ngram_max=ngram_max,
        min_df=min_df,
        max_df=max_df,
        use_stemming=use_stem,
        remove_stopwords=remove_stop,
        sublinear=sublinear,
        norm=norm,
    )

    # Ajuste con documentos
    X = vec.fit_transform(docs)            # (n_docs, n_terms)
    feature_names = np.array(vec.get_feature_names_out())

    # Vector de la pregunta
    q_vec = vec.transform([question])      # (1, n_terms)

    # Similitud coseno
    sims = cosine_similarity(q_vec, X).flatten()
    order = np.argsort(-sims)

    # ------------------- Vista de similitud por doc -------------------
    st.markdown("### 📈 Similitud por documento")
    sim_table = pd.DataFrame({
        "Documento": [f"Doc {i+1}" for i in range(len(docs))],
        "Texto": docs,
        "Similitud": sims
    }).sort_values("Similitud", ascending=False)
    st.dataframe(sim_table.style.format({"Similitud": "{:.3f}"}), use_container_width=True, hide_index=True)

    # Mejor documento
    best_idx = int(order[0])
    best_doc = docs[best_idx]
    best_score = float(sims[best_idx])

    # ------------------- Matriz TF-IDF (recortada) -------------------
    st.markdown("### 📊 Matriz TF-IDF")
    Xdense = X.toarray()
    if show_all_tokens:
        df_tfidf = pd.DataFrame(
            Xdense, columns=feature_names, index=[f"Doc {i+1}" for i in range(len(docs))]
        )
        st.dataframe(df_tfidf.round(3), use_container_width=True)
    else:
        # Elegir Top-K términos por varianza global para una vista compacta
        variances = Xdense.var(axis=0)
        topk_idx = np.argsort(-variances)[:topk_matrix]
        df_tfidf = pd.DataFrame(
            Xdense[:, topk_idx],
            columns=feature_names[topk_idx],
            index=[f"Doc {i+1}" for i in range(len(docs))]
        )
        st.dataframe(df_tfidf.round(3), use_container_width=True)

    # ------------------- Términos relevantes del mejor doc -------------------
    st.markdown("### 🏷️ Términos más relevantes del documento ganador")
    row = X[best_idx, :].toarray().ravel()
    top_doc_idx = np.argsort(-row)[:10]
    top_doc_terms = feature_names[top_doc_idx]
    top_doc_scores = row[top_doc_idx]
    df_top_terms = pd.DataFrame({"Término": top_doc_terms, "TF-IDF": top_doc_scores})
    st.dataframe(df_top_terms.style.format({"TF-IDF": "{:.3f}"}), use_container_width=True, hide_index=True)

    # ------------------- Resaltado de la respuesta -------------------
    # Derivar términos de la pregunta (tokenizados) para resaltar
    q_terms = tokenize_es(question, use_stemming=use_stem, remove_stopwords=remove_stop)
    # Conserva solo términos que estén en el vocabulario (mejor señal)
    vocab_set = set(feature_names)
    to_highlight = [t for t in q_terms if t in vocab_set]
    highlighted = highlight_terms(best_doc, to_highlight)

    st.markdown("### 🎯 Respuesta")
    st.markdown(f"**Tu pregunta:** {question}")
    if best_score > 0.05:
        st.success(f"**Documento más similar (Doc {best_idx+1})** — similitud: **{best_score:.3f}**")
    else:
        st.warning(f"**Documento más similar (Doc {best_idx+1})** — similitud baja: **{best_score:.3f}**")

    st.markdown(f"<div style='padding:.6rem 1rem;border-radius:12px;border:1px solid #e5e7eb;background:#f8fafc'>{highlighted}</div>", unsafe_allow_html=True)

    # ------------------- Explicación breve -------------------
    with st.expander("ℹ️ ¿Cómo funciona este demo?"):
        st.markdown("""
- **Limpieza** del texto → minúsculas, sin URLs/acentos/signos.
- **Tokenización** (con *stemming* opcional) y eliminación de **stopwords**.
- Construimos un espacio TF-IDF con **n-gramas**.
- Calculamos **similitud coseno** entre tu pregunta y cada documento.
- Mostramos el documento con mayor similitud y los **términos TF-IDF** más fuertes.
        """)

