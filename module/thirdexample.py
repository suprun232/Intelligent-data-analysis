import numpy as np, os, matplotlib.pyplot as plt
np.random.seed(42)
os.makedirs("figs", exist_ok=True)


def savefig(fig, name):
    path = os.path.join("figs", name); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig); return path


# Геометрія TSP
n = 20
coords = np.random.rand(n,2)
D = np.sqrt(((coords[:,None,:]-coords[None,:,:])**2).sum(-1))+1e-12
np.fill_diagonal(D, 1e9)  # заборона самопереходу


# Параметри ACO
alpha, beta, rho, Q = 1.0, 3.0, 0.5, 1.0
n_ants, n_iter = n, 60
tau = np.ones_like(D)


def tour_length(path):
    return D[path, np.roll(path,-1)].sum()


best_len, best_path = np.inf, None


for it in range(n_iter):
    all_paths = []
    all_lens = []
    for a in range(n_ants):
        path = [np.random.randint(n)]
        unvisited = set(range(n)); unvisited.remove(path[0])
        while unvisited:
            i = path[-1]
            U = np.array(sorted(list(unvisited)))
            # Ймовірності переходів
            prob = (tau[i,U]**alpha) * ((1.0/D[i,U])**beta)
            prob = prob/prob.sum()
            j = np.random.choice(U, p=prob)
            path.append(j); unvisited.remove(j)
        path = np.array(path, int)
        L = tour_length(path)
        all_paths.append(path); all_lens.append(L)
    # Оновлення найкращого
    idx = int(np.argmin(all_lens))
    if all_lens[idx] < best_len:
        best_len, best_path = all_lens[idx], all_paths[idx]
    # Оновлення феромонів
    tau *= (1 - rho)
    # Глобальне підсилення лише найкращого
    bp = best_path
    for i,j in zip(bp, np.roll(bp,-1)):
        tau[i,j] += Q / best_len
        tau[j,i] += Q / best_len


# Візуалізація найкращого туру
fig = plt.figure(figsize=(4,4))
x,y = coords[:,0], coords[:,1]
plt.scatter(x,y)
for i,j in zip(best_path, np.roll(best_path,-1)):
    plt.plot([x[i],x[j]], [y[i],y[j]])
plt.title(f"ACO TSP (n={n}), best length ~ {best_len:.3f}")
plt.axis("equal"); plt.tight_layout()
fname = savefig(fig, "tsp_route.png")
print("[FIG: tsp_route.png]")
