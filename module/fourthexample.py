import re, os, numpy as np, matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


os.makedirs("figs", exist_ok=True)
def savefig(fig, name):
    path = os.path.join("figs", name); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig); return path


corpus = [
    "Машинне навчання допомагає будувати моделі та робити прогнози.",
    "Глибоке навчання використовує нейронні мережі для складних задач.",
    "Оптимізація параметрів покращує якість моделі і узагальнення.",
    "Нейронні мережі та регуляризація зменшують перенавчання.",
    "Відбір ознак і нормалізація впливають на стабільність і точність.",
    "Енсамблеві методи, як випадковий ліс, підвищують надійність."
]


# TF-IDF
vectorizer = TfidfVectorizer(token_pattern=r"[А-Яа-яA-Za-zЇїІіЄєҐґ-]{2,}", lowercase=True)
X = vectorizer.fit_transform(corpus)
tfidf = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
print(tfidf.round(3).head())


# Zipf: частоти по корпусу
tokens = []
for doc in corpus:
    tokens += re.findall(r"[А-Яа-яA-Za-zЇїІіЄєҐґ-]{2,}", doc.lower())
freq = Counter(tokens)
counts = np.array(sorted(freq.values(), reverse=True))
ranks = np.arange(1, len(counts)+1)


# Лог-лог графік
fig = plt.figure(figsize=(4,3))
plt.loglog(ranks, counts, marker='o')
plt.xlabel("rank"); plt.ylabel("frequency")
plt.title("Zipf plot (лог-лог)")
plt.tight_layout()
fname = savefig(fig, "zipf_loglog.png")
print("[FIG: zipf_loglog.png]")