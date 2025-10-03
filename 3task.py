#1
import numpy as np

rng = np.random.default_rng(42)

# Синтетичні 2D-дані з 3 хмар
centers = np.array([[0, 0], [5, 5], [-5, 5]])
X = np.vstack([
    rng.normal(centers[0], 0.8, size=(200, 2)),
    rng.normal(centers[1], 0.9, size=(200, 2)),
    rng.normal(centers[2], 0.7, size=(200, 2)),
])

def kmeans_pp_init(X, k, rng):
    n = X.shape[0]
    centroids = np.empty((k, X.shape[1]))
    # 1) Обери перший центроїд випадково
    idx = rng.integers(0, n)
    centroids[0] = X[idx]
    # 2) Далі - kmeans++ відбір
    for i in range(1, k):
        d2 = np.min(np.sum((X[:, None, :] - centroids[None, :i, :])**2, axis=2), axis=1)
        probs = d2 / d2.sum()
        idx = rng.choice(n, p=probs)
        centroids[i] = X[idx]
    return centroids

def kmeans(X, k, max_iter=100, tol=1e-4, n_init=5, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    best_inertia = np.inf
    best = None
    for _ in range(n_init):
        C = kmeans_pp_init(X, k, rng)
        for _ in range(max_iter):
            # Призначення
            d2 = np.sum((X[:, None, :] - C[None, :, :])**2, axis=2)  # n x k
            labels = np.argmin(d2, axis=1)
            # Оновлення
            C_new = np.vstack([
                X[labels == j].mean(axis=0) if np.any(labels == j) else C[j]
                for j in range(k)
            ])
            shift = np.linalg.norm(C_new - C)
            C = C_new
            if shift < tol:
                break
        inertia = np.sum((X - C[labels])**2)
        if inertia < best_inertia:
            best_inertia = inertia
            best = (C.copy(), labels.copy(), inertia)
    return best  # (centroids, labels, inertia)

C, labels, inertia = kmeans(X, k=3, n_init=10, rng=rng)
print("Centroids:\n", C)
print("Inertia (SSE):", inertia)
#2

import numpy as np

class SOM:
    def __init__(self, m, n, dim, lr=0.5, sigma=None, rng=None):
        self.m, self.n, self.dim = m, n, dim
        self.lr0 = lr
        self.sigma0 = sigma if sigma is not None else max(m, n) / 2
        self.rng = rng or np.random.default_rng()
        self.W = self.rng.normal(0, 1, size=(m, n, dim))
        # Координати решітки
        self.grid_y, self.grid_x = np.meshgrid(np.arange(m), np.arange(n), indexing='ij')

    def _bmu(self, x):
        # Повертає індекси BMU для вектора x
        d2 = np.sum((self.W - x)**2, axis=2)
        iy, ix = np.unravel_index(np.argmin(d2), (self.m, self.n))
        return iy, ix

    def fit(self, X, epochs=20):
        T = epochs * len(X)
        t = 0
        for epoch in range(epochs):
            self.rng.shuffle(X)
            for x in X:
                # Параметри, що спадають
                lr = self.lr0 * np.exp(-t / T)
                sigma = self.sigma0 * np.exp(-t / T)
                # BMU
                iy, ix = self._bmu(x)
                # Радіус сусідства
                d2 = (self.grid_y - iy)**2 + (self.grid_x - ix)**2
                h = np.exp(-d2 / (2 * (sigma**2) + 1e-8))  # m x n
                # Оновлення ваг
                self.W += lr * h[..., None] * (x - self.W)
                t += 1

    def transform(self, X):
        # Повертає координати BMU на решітці для кожної точки
        coords = []
        for x in X:
            coords.append(self._bmu(x))
        return np.array(coords)

# Дані: 3 кластери (як вище)
rng = np.random.default_rng(0)
centers = np.array([[0, 0], [5, 5], [-5, 5]])
X = np.vstack([
    rng.normal(centers[0], 0.8, size=(200, 2)),
    rng.normal(centers[1], 0.9, size=(200, 2)),
    rng.normal(centers[2], 0.7, size=(200, 2)),
])

som = SOM(m=10, n=10, dim=2, lr=0.5, rng=rng)
som.fit(X, epochs=15)
bmu_coords = som.transform(X)

# Підрахунок, скільки точок "припало" на кожен нейрон
heat = np.zeros((som.m, som.n), dtype=int)
for (iy, ix) in bmu_coords:
    heat[iy, ix] += 1

print("Top-5 найзаповненіших нейронів (iy, ix, count):")
hotspots = np.dstack(np.unravel_index(np.argsort(heat, axis=None)[::-1], heat.shape))[0]
for k in range(5):
    iy, ix = hotspots[k]
    print((iy, ix, int(heat[iy, ix])))

#3


import numpy as np

rng = np.random.default_rng(123)

# Випадкові міста на площині
n_cities = 15
coords = rng.random((n_cities, 2)) * 100.0  # (x, y) у [0, 100]

# Евклідова відстань
D = np.sqrt(((coords[:, None, :] - coords[None, :, :])**2).sum(axis=2)) + 1e-9  # уникаємо 0
eta = 1.0 / D  # евристика

# Параметри ACO
alpha = 1.0        # вага феромонів
beta = 5.0         # вага евристики
rho = 0.5          # випаровування
Q = 100.0          # інтенсивність підсилення
n_ants = n_cities  # зазвичай ~ кількості міст
n_iters = 150

tau = np.full((n_cities, n_cities), 1.0)  # початковий феромон
best_len = np.inf
best_tour = None

def tour_length(tour):
    # повний цикл + повернення
    return sum(D[tour[i], tour[(i+1) % len(tour)]] for i in range(len(tour)))

for it in range(n_iters):
    all_tours = []
    all_lens = []

    for a in range(n_ants):
        start = rng.integers(0, n_cities)
        unvisited = set(range(n_cities))
        unvisited.remove(start)
        tour = [start]
        current = start

        while unvisited:
            J = list(unvisited)
            # ймовірності переходу
            numerators = (tau[current, J]**alpha) * (eta[current, J]**beta)
            p = numerators / numerators.sum()
            nxt = rng.choice(J, p=p)
            tour.append(nxt)
            unvisited.remove(nxt)
            current = nxt

        L = tour_length(tour)
        all_tours.append(tour)
        all_lens.append(L)

    # Оновлення глобальних феромонів
    tau *= (1.0 - rho)
    # Підсилюємо тільки найкращий тур цієї ітерації (можна всі — класичний Ant System)
    idx_best = int(np.argmin(all_lens))
    iter_best_tour = all_tours[idx_best]
    iter_best_len = all_lens[idx_best]

    # Депозит феромонів уздовж ребер найкращого туру
    deposit = Q / iter_best_len
    for i in range(n_cities):
        u = iter_best_tour[i]
        v = iter_best_tour[(i + 1) % n_cities]
        tau[u, v] += deposit
        tau[v, u] += deposit

    if iter_best_len < best_len:
        best_len = iter_best_len
        best_tour = iter_best_tour

print("Найкраща знайдена довжина туру:", best_len)
print("Тур:", best_tour)
