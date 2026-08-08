"""
preprocessing.py
----------------
Lógica de limpieza (Regex + normalización) y lematización con SpaCy.
"""

import html
import re
import spacy

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# El corpus AG News trae entidades HTML "rotas": les falta el "&" inicial
# (p. ej. "quot;", "#39;", "amp;" en vez de "&quot;", "&#39;", "&amp;").
# html.unescape() por sí solo NO las detecta porque no son válidas sin el "&".
_NAMED_ENTITIES = ["quot", "amp", "lt", "gt", "apos", "nbsp"]
_NAMED_RE = re.compile(r"&?(" + "|".join(_NAMED_ENTITIES) + r");", re.IGNORECASE)
_NUMERIC_RE = re.compile(r"&?#(\d+);")


def clean_text(text: str) -> str:
    """
    Limpieza a nivel de string, vía Regex, antes de tokenizar.
    - Decodifica entidades HTML (&lt;, &amp;, &quot;, etc.) a su forma real para que los tags que codifican queden expuestos y se puedan eliminar (si no, "&lt;FONT&gt;" sobrevive como ruido "lt font gt" en el corpus)
    - Elimina tags HTML residuales (ya decodificados o literales)
    - Elimina URLs
    - Elimina caracteres que no sean letras (números, puntuación, símbolos)
    - Normaliza espacios múltiples
    - Pasa todo a minúsculas
    """
    text = str(text)
    text = _NUMERIC_RE.sub(lambda m: chr(int(m.group(1))), text)   # "#39;" -> "'"
    text = _NAMED_RE.sub(lambda m: html.unescape(f"&{m.group(1)};"), text)  # "quot;" -> '"'
    text = html.unescape(text)                              # &lt;FONT&gt; -> <FONT> (por si vienen bien formadas)
    text = re.sub(r"<[^>]+>", " ", text)                    # tags HTML
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # URLs
    text = re.sub(r"[^a-zA-Z\s]", " ", text)                 # solo letras
    text = re.sub(r"\s+", " ", text).strip()                  # espacios extra
    return text.lower()


def preprocess_text(text: str) -> str:

    cleaned = clean_text(text)
    doc = nlp(cleaned)

    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop and not token.is_punct and not token.is_space
        and len(token.lemma_) > 1
    ]

    return " ".join(tokens)


def preprocess_corpus(texts, batch_size: int = 64, n_process: int = 1):
    """
    Versión batch de preprocess_text, usando nlp.pipe para procesar todo un corpus de forma eficiente (en vez de doc-por-doc). Aplica exactamente la misma lógica que preprocess_text.
    """
    cleaned_texts = [clean_text(t) for t in texts]
    results = []

    for doc in nlp.pipe(cleaned_texts, batch_size=batch_size, n_process=n_process):
        tokens = [
            token.lemma_
            for token in doc
            if not token.is_stop and not token.is_punct and not token.is_space
            and len(token.lemma_) > 1
        ]
        results.append(" ".join(tokens))

    return results
