import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

# --- тестова функція: Розенброк (2D) ---
def f(x):
    x1, x2 = x
    return 100.0*(x2 - x1**2)**2 + (1 - x1)**2

def g(x):
    x1, x2 = x
    return np.array([
        -400*x1*(x2 - x1**2) - 2*(1 - x1),
         200*(x2 - x1**2)
    ], dtype=float)

# --- оптимізатори ---
def gd(x0, lr=1e-3, iters=5000):
    x = x0.copy()
    hist = []
    for k in range(iters):
        grad = g(x)
        # clip для стабільності
        if np.linalg.norm(grad) > 100:
            grad = grad / np.linalg.norm(grad) * 100
        x -= lr * grad
        hist.append(f(x))
        if np.linalg.norm(grad) < 1e-6:
            break
    return x, np.array(hist), k+1

def momentum(x0, lr=1e-3, beta=0.9, iters=5000):
    x = x0.copy()
    v = np.zeros_like(x)
    hist = []
    for k in range(iters):
        grad = g(x)
        if np.linalg.norm(grad) > 100:
            grad = grad / np.linalg.norm(grad) * 100
        v = beta * v + grad
        x -= lr * v
        hist.append(f(x))
        if np.linalg.norm(grad) < 1e-6:
            break
    return x, np.array(hist), k+1

def nesterov(x0, lr=1e-3, beta=0.9, iters=5000):
    x = x0.copy()
    v = np.zeros_like(x)
    hist = []
    for k in range(iters):
        grad = g(x - beta * v)
        if np.linalg.norm(grad) > 100:
            grad = grad / np.linalg.norm(grad) * 100
        v = beta * v + grad
        x -= lr * v
        hist.append(f(x))
        if np.linalg.norm(g(x)) < 1e-6:
            break
    return x, np.array(hist), k+1

def adam(x0, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, iters=5000):
    x = x0.copy()
    m = np.zeros_like(x); v = np.zeros_like(x)
    hist = []
    for t in range(1, iters+1):
        grad = g(x)
        if np.linalg.norm(grad) > 100:
            grad = grad / np.linalg.norm(grad) * 100
        m = b1*m + (1-b1)*grad
        v = b2*v + (1-b2)*(grad*grad)
        mhat = m / (1-b1**t)
        vhat = v / (1-b2**t)
        x -= lr * mhat / (np.sqrt(vhat) + eps)
        hist.append(f(x))
        if np.linalg.norm(grad) < 1e-6:
            break
    return x, np.array(hist), t


def run():
    x0 = np.array([-1.2, 1.0])  # стандартний старт для Rosenbrock
    configs = [
        ("GD",       lambda: gd(x0, lr=1e-3, iters=20000)),
        ("Momentum", lambda: momentum(x0, lr=2e-3, beta=0.9, iters=20000)),
        ("Nesterov", lambda: nesterov(x0, lr=2e-3, beta=0.9, iters=20000)),
        ("Adam",     lambda: adam(x0, lr=5e-3, iters=20000)),
    ]

    results = []
    for name, fn in configs:
        x_star, hist, iters = fn()
        results.append((name, x_star, hist, iters))

    # друк короткої таблиці
    print(f"{'Method':10s} {'f(x)':>12s} {'||grad||':>12s} {'Iters':>8s}")
    print("-"*46)
    for name, x_star, hist, iters in results:
        fx = f(x_star); gn = np.linalg.norm(g(x_star))
        print(f"{name:10s} {fx:12.4e} {gn:12.3e} {iters:8d}")

    # графік збіжності
    plt.figure(figsize=(7,5))
    for name, _, hist, _ in results:
        vals = hist - np.min(hist)
        plt.semilogy(vals + 1e-16, label=name)  # лог-шкала
    plt.xlabel("Ітерація")
    plt.ylabel("f(x) - min f (зсунуто)")
    plt.title("Порівняння градієнтних методів на Rosenbrock(2D)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("gradient_methods_convergence.png", dpi=150)
    print("\nЗбережено: gradient_methods_convergence.png")

if __name__ == "__main__":
    run()
