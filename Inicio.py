# app.py — Demo TF-IDF en Español (actualizado con fixes min_df/max_df y validaciones)
import re
import unicodedata
from typing import List

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Opcional: stemming español (fallback si no está NLTK)
try:
    from nltk.stem import SnowballStemmer
    STEMMER = SnowballStemmer("spanish")
    def stem_es(t: str) -> str: return STEMMER.stem(t)
except Exception:
    def stem_es(t: str) -> str: return t

st.set_page_config(page_title="Demo TF-IDF en Español", page_icon="🔍", layout="wide")
st.title("🔍 Demo TF-IDF en Español – Laboratorio")

# ---------------------------------------------------------------------
# Datos de ejemplo
# ---------------------------------------------------------------------
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

STOPWORDS_ES = set("""
a al algo algunas algunos ante antes así aún cada como con contra cuándo cuál cuáles cuando de del desde donde dos el ella ellas ellos en entre era erais eran eras eres es esa esas ese eso esos esta estaban estado estaba estáis estamos están estar estará estas este éstos esto estos estoy fue fuera fueron fui fuimos ha habían había habíais habían habíamos haber hace hacen hacer hacia han hasta hay la las le les lo los más me mi mis mucha muchas mucho muchos muy nada ni no nos nosotras nosotros o os otra otras otro otros para pero poco por porque qué que quien quién quienes se ser será si sido sin sobre sois somos son soy su sus también te tiene tienen tener tengo tenía tenían tenemos tenéis tus tu un una uno unas unos ya
""".split())

# ---------------------------------------------------------------------
# Utils de limpieza/tokenización
# ---------------------------------------------------------------------
def strip_accents(s: str) -> str:
    nkfd = unicodedata.normalize("NFKD", s)
    return "".join([c for c in nkfd if not unicodedata.combining(c)])

def tokenize_es(text: str, use_stemming: bool, remove_stopwords: bool) -> List[str]:
    text = text.lower()
    text = re.sub(r"https?://\S+|[@#]\w+", " ", text)
    text = re.sub(r"[^a-záéíóúüñ\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = strip_accents(text)
    tokens = [t for t in text.split() if len(t) > 1]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS_ES]
    if use_stemming:
        tokens = [stem_es(t) for t in tokens]
    return tokens

def build_vectorizer(ngram_max: int, min_df, max_df,
                     use_stemming: bool, remove_stopwords: bool,
                     sublinear: bool, norm):
    return TfidfVectorizer(
        tokenizer=lambda s: tokenize_es(s, use_stemming=use_stemming, remove_stopwords=remove_stopwords),
        ngram_range=(1, ngram_max),
        min_df=min_df,             # puede ser int (conteo) o float (proporción)
        max_df=max_df,             # idem
        use_idf=True,
        sublinear_tf=sublinear,
        norm=norm,
    )

def highlight_terms(text: str, terms: List[str]) -> str:
    if not terms: return text
    safe = [re.escape(t) for t in sorted(set(terms), key=len, reverse=True)]
    if not safe: return text
    pattern = r"\b(" + "|".join(safe) + r")\b"
    return re.sub(pattern, lambda m: f"<mark>{m.group(0)}</mark>", text, flags=re.IGNORECASE)

# ---------------------------------------------------------------------
# Sidebar — Parámetros (incluye modo de df para evitar ValueError)
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Parámetros del vectorizador")

    ngram_max = st.select_slider("N-gramas (máx)", options=[1, 2, 3], value=2)

    mode_df = st.radio("Modo de df", ["Conteo (int)", "Proporción (0–1)"], index=0, horizontal=True)
    if mode_df == "Conteo (int)":
        min_df_ui = st.number_input("min_df (conteo)", value=1, min_value=1, step=1,
                                    help="≥1 = al menos ese # de documentos. Recomendado: 1–2.")
        max_df_ui = st.number_input("max_df (conteo)", value=1000, min_value=1, step=1,
                                    help="Filtra términos demasiado frecuentes por conteo. Déjalo alto para no filtrar.")
        min_df = int(min_df_ui)
        max_df = int(max_df_ui)
    else:
        min_df_ui = st.number_input("min_df (proporción 0–1)", value=0.0, min_value=0.0, max_value=1.0, step=0.05,
                                    help="0.0–1.0. 0.0 = sin filtro inferior. Evita 1.0 (100%).")
        max_df_ui = st.number_input("max_df (proporción 0–1)", value=1.0, min_value=0.0, max_value=1.0, step=0.05,
                                    help="Filtra términos muy frecuentes. 1.0 = sin filtro superior.")
        min_df = float(min_df_ui)
        max_df = float(max_df_ui)

    remove_stop = st.checkbox("Quitar stopwords", True)
    use_stem = st.checkbox("Aplicar stemming (Snowball)", True)
    sublinear = st.checkbox("Sublinear TF (log(1+tf))", True)
    norm = st.selectbox("Normalización", ["l2", None], index=0)

    st.markdown("---")
    topk_matrix = st.slider("Top-K términos para mostrar en matriz", 5, 50, 20, 1)
    show_all_tokens = st.checkbox("Mostrar TODAS las columnas (puede ser grande)", False)

# ---------------------------------------------------------------------
# Layout principal
# ---------------------------------------------------------------------
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

if "question" in st.session_state:
    question = st.session_state.question

# ---------------------------------------------------------------------
# Botón Analizar
# ---------------------------------------------------------------------
if st.button("🔎 Analizar", type="primary"):
    docs = [d.strip() for d in text_input.split("\n") if d.strip()]
    if len(docs) < 1:
        st.error("⚠️ Ingresa al menos un documento.")
        st.stop()
    if not question.strip():
        st.error("⚠️ Escribe una pregunta.")
        st.stop()

    # Diagnóstico: asegúrate que no queden docs vacíos tras tokenización
    token_lists = [tokenize_es(d, use_stemming=use_stem, remove_stopwords=remove_stop) for d in docs]
    empty_docs = [i for i, toks in enumerate(token_lists) if len(toks) == 0]
    if empty_docs:
        st.error(
            "Estos documentos quedaron VACÍOS tras la limpieza/tokenización: "
            + ", ".join([f"Doc {i+1}" for i in empty_docs])
            + ". Reduce filtros, desactiva stopwords/stemming o revisa el texto."
        )
        st.stop()

    vec = build_vectorizer(
        ngram_max=ngram_max,
        min_df=min_df,
        max_df=max_df,
        use_stemming=use_stem,
        remove_stopwords=remove_stop,
        sublinear=sublinear,
        norm=norm,
    )

    # Ajuste con manejo del ValueError típico de pruning
    try:
        X = vec.fit_transform(docs)   # (n_docs, n_terms)
    except ValueError:
        st.error("No quedaron términos después del pruning de TF-IDF.")
        st.info(
            "- Usa **min_df=1** (conteo) o baja la proporción (p.ej., 0.0–0.1).\n"
            "- Sube **max_df** (p.ej., 1.0 o un conteo grande).\n"
            "- Desactiva temporalmente **stopwords** y/o **stemming**.\n"
            "- Amplía **n-gramas** (2 o 3)."
        )
        st.stop()

    feature_names = np.array(vec.get_feature_names_out())
    q_vec = vec.transform([question])
    sims = cosine_similarity(q_vec, X).flatten()
    order = np.argsort(-sims)

    # --- Similitud por documento
    st.markdown("### 📈 Similitud por documento")
    sim_table = pd.DataFrame({
        "Documento": [f"Doc {i+1}" for i in range(len(docs))],
        "Texto": docs,
        "Similitud": sims
    }).sort_values("Similitud", ascending=False)
    st.dataframe(sim_table.style.format({"Similitud": "{:.3f}"}), use_container_width=True, hide_index=True)

    best_idx = int(order[0])
    best_doc = docs[best_idx]
    best_score = float(sims[best_idx])

    # --- Matriz TF-IDF (recortada)
    st.markdown("### 📊 Matriz TF-IDF")
    Xdense = X.toarray()
    if show_all_tokens:
        df_tfidf = pd.DataFrame(
            Xdense, columns=feature_names, index=[f"Doc {i+1}" for i in range(len(docs))]
        )
        st.dataframe(df_tfidf.round(3), use_container_width=True)
    else:
        variances = Xdense.var(axis=0)
        topk_idx = np.argsort(-variances)[:topk_matrix]
        df_tfidf = pd.DataFrame(
            Xdense[:, topk_idx],
            columns=feature_names[topk_idx],
            index=[f"Doc {i+1}" for i in range(len(docs))]
        )
        st.dataframe(df_tfidf.round(3), use_container_width=True)

    # --- Términos top del documento ganador
    st.markdown("### 🏷️ Términos más relevantes del documento ganador")
    row = X[best_idx, :].toarray().ravel()
    top_doc_idx = np.argsort(-row)[:10]
    top_doc_terms = feature_names[top_doc_idx]
    top_doc_scores = row[top_doc_idx]
    df_top_terms = pd.DataFrame({"Término": top_doc_terms, "TF-IDF": top_doc_scores})
    st.dataframe(df_top_terms.style.format({"TF-IDF": "{:.3f}"}), use_container_width=True, hide_index=True)

    # --- Resaltado de la respuesta
    q_terms = tokenize_es(question, use_stemming=use_stem, remove_stopwords=remove_stop)
    vocab_set = set(feature_names)
    to_highlight = [t for t in q_terms if t in vocab_set]
    highlighted = highlight_terms(best_doc, to_highlight)

    st.markdown("### 🎯 Respuesta")
    st.markdown(f"**Tu pregunta:** {question}")
    if best_score > 0.05:
        st.success(f"**Documento más similar (Doc {best_idx+1})** — similitud: **{best_score:.3f}**")
    else:
        st.warning(f"**Documento más similar (Doc {best_idx+1})** — similitud baja: **{best_score:.3f}**")
    st.markdown(
        f"<div style='padding:.6rem 1rem;border-radius:12px;border:1px solid #e5e7eb;background:#f8fafc'>{highlighted}</div>",
        unsafe_allow_html=True
    )

    with st.expander("ℹ️ ¿Cómo funciona este demo?"):
        st.markdown("""
- **Limpieza** del texto → minúsculas, sin URLs/acentos/signos.
- **Tokenización** (con *stemming* opcional) y eliminación de **stopwords**.
- Espacio TF-IDF con **n-gramas** configurables.
- **Similitud coseno** entre tu pregunta y cada documento.
- Documento con mayor similitud + **términos TF-IDF** más fuertes.
        """)
