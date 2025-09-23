import numpy as np
import math
import time
import matplotlib.pyplot as plt

# -----------------------------
# Загальний шаблон нелінійного CG
# -----------------------------
def backtracking_armijo(f, g, x, d, f0=None, g0=None, alpha0=1.0, c1=1e-4, tau=0.5, max_back=30):
    """
    Проста лінійна пошукова процедура (Armijo) без умови кривизни.
    Повертає alpha, f(x+alpha d), g(x+alpha d), кількість перевірок.
    """
    if f0 is None: f0 = f(x)
    if g0 is None: g0 = g(x)
    dg = np.dot(g0, d)
    alpha = alpha0
    nfev = 0
    for _ in range(max_back):
        x_new = x + alpha * d
        f_new = f(x_new); nfev += 1
        if f_new <= f0 + c1 * alpha * dg:
            g_new = g(x_new)
            return alpha, f_new, g_new, nfev
        alpha *= tau
    # якщо не вдалося — приймаємо дуже малий крок
    g_new = g(x + alpha * d)
    return alpha, f(x + alpha * d), g_new, nfev

def nlcg(
    f, g, x0, beta_rule="FR", tol=1e-6, max_iter=2000,
    line_search="armijo", alpha0=1.0, restart="powell", verbose=False
):
    """
    Нелінійний спряжений градієнт з різними формулами бета.
    beta_rule ∈ {"FR","PRP+","HS+","DY"}.
    restart: None або "powell" (перезапуск, якщо d_k^T g_k >= 0).
    """
    x = x0.copy().astype(float)
    fval = f(x); grad = g(x)
    d = -grad
    it, nfev, ngev = 0, 1, 1
    hist_f, hist_grad = [fval], [np.linalg.norm(grad)]

    while it < max_iter and np.linalg.norm(grad) > tol:
        # простий вибір початкового кроку
        alpha_guess = alpha0 if it == 0 else min(1.0, 2.0*alpha0)
        alpha, fnew, gnew, fev = backtracking_armijo(f, g, x, d, f0=fval, g0=grad, alpha0=alpha_guess)
        nfev += fev; ngev += 1
        x_new = x + alpha * d

        y = gnew - grad
        if beta_rule == "FR":
            beta = np.dot(gnew, gnew) / (np.dot(grad, grad) + 1e-16)
        elif beta_rule == "PRP+":
            beta_raw = np.dot(gnew, y) / (np.dot(grad, grad) + 1e-16)
            beta = max(beta_raw, 0.0)
        elif beta_rule == "HS+":
            denom = np.dot(d, y) + 1e-16
            beta_raw = np.dot(gnew, y) / denom
            beta = max(beta_raw, 0.0)
        elif beta_rule == "DY":
            denom = np.dot(d, y) + 1e-16
            beta = np.dot(gnew, gnew) / denom
        else:
            raise ValueError("Unknown beta rule")

        # напрямок
        d_new = -gnew + beta * d

        # перезапуск за Пауеллом (підвищує стабільність)
        if restart == "powell" and np.dot(d_new, gnew) >= 0:
            d_new = -gnew

        x, fval, grad, d = x_new, fnew, gnew, d_new
        it += 1
        hist_f.append(fval)
        hist_grad.append(np.linalg.norm(grad))
        if verbose and it % 50 == 0:
            print(f"[{beta_rule}] iter {it:4d}  f={fval:.6e}  ||g||={hist_grad[-1]:.3e}  alpha={alpha:.2e}")

    return {
        "x": x,
        "f": fval,
        "g_norm": np.linalg.norm(grad),
        "iters": it,
        "nfev": nfev,
        "ngev": ngev,
        "hist_f": np.array(hist_f),
        "hist_grad": np.array(hist_grad),
        "converged": np.linalg.norm(grad) <= tol
    }

# -----------------------------
# Тестові задачі оптимізації
# -----------------------------
def quad_problem(dim=50, cond=1e4, seed=0):
    rng = np.random.default_rng(seed)
    # Діагональна SPD матриця з заданою обумовленістю
    vals = np.logspace(0, math.log10(cond), dim)
    A = np.diag(vals)
    # розв'язок x* = A^{-1} b (виберемо довільний x*, під нього побудуємо b)
    x_star = rng.normal(size=dim)
    b = A @ x_star
    def f(x): return 0.5 * x @ (A @ x) - b @ x
    def g(x): return A @ x - b
    x0 = rng.normal(size=dim)
    return f, g, x0, x_star

def rosenbrock(n=2):
    assert n >= 2
    def f(x):
        x = np.asarray(x)
        return np.sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)
    def g(x):
        x = np.asarray(x)
        grad = np.zeros_like(x)
        grad[:-1] = -400*x[:-1]*(x[1:]-x[:-1]**2) - 2*(1-x[:-1])
        grad[1:] += 200*(x[1:]-x[:-1]**2)
        return grad
    x0 = np.array([-1.2, 1.0] + [0.5]*(n-2), dtype=float)
    x_star = np.ones(n)
    return f, g, x0, x_star

# -----------------------------
# Порівняння методів
# -----------------------------
def run_suite():
    methods = ["FR", "PRP+", "HS+", "DY"]
    results = []

    # 1) Квадратична задача (погано обумовлена)
    f, g, x0, x_star = quad_problem(dim=80, cond=1e6, seed=1)
    for m in methods:
        t0 = time.time()
        res = nlcg(f, g, x0, beta_rule=m, tol=1e-8, max_iter=5000, alpha0=1.0, restart="powell")
        t1 = time.time()
        res.update(problem="Quadratic(cond=1e6)", method=m, time=t1-t0, x_star_norm=np.linalg.norm(x_star))
        results.append(res)

    # 2) Розенброк 2D
    f, g, x0, x_star = rosenbrock(n=2)
    for m in methods:
        t0 = time.time()
        res = nlcg(f, g, x0, beta_rule=m, tol=1e-6, max_iter=5000, alpha0=1.0, restart="powell")
        t1 = time.time()
        res.update(problem="Rosenbrock(2D)", method=m, time=t1-t0, x_star_norm=np.linalg.norm(x_star))
        results.append(res)

    # 3) Розширений Розенброк 10D
    f, g, x0, x_star = rosenbrock(n=10)
    for m in methods:
        t0 = time.time()
        res = nlcg(f, g, x0, beta_rule=m, tol=1e-6, max_iter=10000, alpha0=1.0, restart="powell")
        t1 = time.time()
        res.update(problem="Rosenbrock(10D)", method=m, time=t1-t0, x_star_norm=np.linalg.norm(x_star))
        results.append(res)

    # Друк підсумкової таблиці
    header = f"{'Problem':20s} {'Method':6s} {'f*':>11s} {'||g||':>11s} {'Iter':>6s} {'f-evals':>7s} {'g-evals':>7s} {'Time(s)':>8s} {'Ok':>3s}"
    print(header)
    print("-"*len(header))
    for r in results:
        print(f"{r['problem']:20s} {r['method']:6s} {r['f']:+.3e} {r['g_norm']:.2e} {r['iters']:6d} {r['nfev']:7d} {r['ngev']:7d} {r['time']:8.3f} {'Y' if r['converged'] else 'N':>3s}")

    # Візуалізація збіжності для Rosenbrock(2D)
    rb = [r for r in results if r["problem"]=="Rosenbrock(2D)"]
    plt.figure(figsize=(7,5))
    for r in rb:
        vals = r["hist_f"]
        vals = vals - vals.min()  # для порівнянності шкали
        plt.semilogy(vals, label=r["method"])
    plt.xlabel("Ітерація")
    plt.ylabel("f(x) - f_min (лог шкала)")
    plt.title("Збіжність на Rosenbrock(2D)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("cg_convergence_rosenbrock2d.png", dpi=150)
    print("\nГрафік збіжності збережено у: cg_convergence_rosenbrock2d.png")

if __name__ == "__main__":
    run_suite()
