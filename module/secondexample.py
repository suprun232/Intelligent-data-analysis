import numpy as np, os, matplotlib.pyplot as plt
np.random.seed(123)
os.makedirs("figs", exist_ok=True)


def savefig(fig, name):
    path = os.path.join("figs", name); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig); return path


N = 50
fitness = np.linspace(0.1, 1.0, N) + 0.1*np.random.rand(N)  # монотонний тренд + шум


def roulette(f):
    p = f / f.sum()
    return np.searchsorted(np.cumsum(p), np.random.rand())


def rank_selection(f, s=1.5):
    idx = np.argsort(f)  # зростання
    ranks = np.empty_like(idx); ranks[idx] = np.arange(N)  # 0 — гірший
    i = ranks  # для кожного індексу його ранг
    p = (2 - s)/N + (2*i*(s - 1))/(N*(N - 1))
    p /= p.sum()
    return np.random.choice(N, p=p)


def tournament(f, k=3):
    cand = np.random.choice(N, size=k, replace=False)
    return cand[np.argmax(f[cand])]


def threshold_sel(f, q=0.8):
    thr = np.quantile(f, q)
    S = np.where(f >= thr)[0]
    if len(S) == 0: return np.argmax(f)
    return np.random.choice(S)


T = 20000
counts = { 'roulette':np.zeros(N,int),
           'rank':np.zeros(N,int),
           'tourn_k3':np.zeros(N,int),
           'thr_q80':np.zeros(N,int) }


for _ in range(T):
    counts['roulette'][roulette(fitness)] += 1
    counts['rank'][rank_selection(fitness, s=1.5)] += 1
    counts['tourn_k3'][tournament(fitness, k=3)] += 1
    counts['thr_q80'][threshold_sel(fitness, q=0.8)] += 1


# Візуалізація: частки вибору vs індекс (1 — найгірший, N — найкращий)
x = np.arange(N)
fig = plt.figure(figsize=(7,3.5))
for name, cnt in counts.items():
    plt.plot(x, cnt / T, label=name)
plt.xlabel("індекс індивіда (0..N-1)"); plt.ylabel("частка вибору")
plt.title("Порівняння операторів відбору (симуляція)")
plt.legend(); plt.tight_layout()
fname = savefig(fig, "selection_hist.png")
print("[FIG: selection_hist.png]")