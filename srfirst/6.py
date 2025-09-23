import numpy as np

# --------- лінійний пошук Armijo із перевіркою допустимості ----------
def armijo_with_feasibility(f, g_list, x, d, f0=None, c1=1e-4, tau=0.5, alpha0=1.0, max_back=40):
    if f0 is None:
        f0 = f(x)
    gradf_dot_d = None  # оцінимо раз на старті через різницю f, якщо треба
    alpha = alpha0
    for _ in range(max_back):
        x_new = x + alpha * d
        # перевірка допустимості (всі g_i <= 0)
        feas = True
        for gi in g_list:
            if gi(x_new) > 0.0:
                feas = False
                break
        if not feas:
            alpha *= tau
            continue
        f_new = f(x_new)
        if gradf_dot_d is None:
            # груба оцінка похідної вздовж d (скінч. різн.)
            eps = 1e-8
            f_eps = f(x + eps * d)
            gradf_dot_d = (f_eps - f0) / eps
        if f_new <= f0 + c1 * alpha * gradf_dot_d:
            return alpha, x_new, f_new
        alpha *= tau
    # якщо все погано — дуже малий крок або нульовий
    return 0.0, x, f0

# --------- проєкційний пошук множників λ ≥ 0 (простий PGD) ----------
def project_nonneg(z):
    return np.maximum(z, 0.0)

def solve_lambda_projected_grad(grad_f, J, max_it=200, lr=1e-1, tol=1e-10):
    """
    Розв'язує min_{λ >= 0} 0.5 || grad_f + J^T λ ||^2
    ∇ = J(grad_f + J^T λ). Простий проєкційний GD по λ.
    Повертає λ і напрямок d = -(grad_f + J^T λ).
    """
    if J.size == 0:
        return np.zeros(0), -grad_f
    m = J.shape[0]
    lam = np.zeros(m)
    for _ in range(max_it):
        r = grad_f + J.T @ lam  # резидуал
        grad_lam = J @ r
        lam_new = project_nonneg(lam - lr * grad_lam)
        if np.linalg.norm(lam_new - lam) < tol:
            lam = lam_new
            break
        lam = lam_new
    d = -(grad_f + J.T @ lam)
    return lam, d

# --------- основний алгоритм Заутендайка (MFD) ----------
def zoutendijk_feasible_directions(
    f, grad_f, g_list, grad_g_list, x0,
    tol=1e-6, max_iter=2000, tau_active=1e-6, alpha0=1.0, verbose=False
):
    """
    f, grad_f: ціль і її градієнт
    g_list, grad_g_list: список обмежень g_i(x) <= 0 та їх градієнтів
    x0: початково допустима точка (важливо!)
    """
    x = x0.copy().astype(float)
    # перевіримо допустимість старту (можна зробити проєкцію, але тут очікуємо feasible x0)
    if any(gi(x) > 0 for gi in g_list):
        raise ValueError("x0 має бути допустимим: деякі g_i(x0) > 0")

    hist = []
    for k in range(1, max_iter + 1):
        gf = grad_f(x)
        # активний набір
        active_idx = [i for i, gi in enumerate(g_list) if gi(x) >= -tau_active]
        J = np.array([grad_g_list[i](x) for i in active_idx], dtype=float) if active_idx else np.zeros((0, x.size))
        lam, d = solve_lambda_projected_grad(gf, J, lr=1e-1)

        # якщо напрямок майже нульовий — кандидат на стаціонарну точку
        if np.linalg.norm(d) < 1e-10:
            if verbose:
                print(f"[k={k}] ‖d‖≈0 → зупинка")
            break

        # лінійний пошук із допустимістю
        f0 = f(x)
        alpha, x_new, f_new = armijo_with_feasibility(f, g_list, x, d, f0=f0, alpha0=alpha0)

        # якщо крок нульовий — також зупиняємось
        if alpha <= 0.0:
            if verbose:
                print(f"[k={k}] alpha=0 → зупинка")
            break

        # протокол
        hist.append(dict(k=k, f=f_new, alpha=alpha, gmax=max(gi(x_new) for gi in g_list), n_active=len(active_idx)))
        if verbose and (k % 20 == 0 or k <= 3):
            print(f"[k={k:4d}] f={f_new:.6e}  alpha={alpha:.2e}  |A|={len(active_idx)}")

        # критерій зупинки по градієнту (проєктованому)
        if np.linalg.norm(d) < tol and abs(f_new - f0) < 1e-12:
            break

        x = x_new

    return {"x": x, "f": f(x), "hist": hist}

# ------------------------ ПРИКЛАДИ ------------------------

# 1) Розенброк з обмеженням кола: x1^2 + x2^2 <= 1.5
def rosenbrock_2d():
    def f(x):
        x1, x2 = x
        return 100*(x2 - x1**2)**2 + (1 - x1)**2
    def grad_f(x):
        x1, x2 = x
        return np.array([-400*x1*(x2 - x1**2) - 2*(1 - x1), 200*(x2 - x1**2)], dtype=float)
    def g1(x):  # коло
        return x[0]**2 + x[1]**2 - 1.5
    def grad_g1(x):
        return 2*np.array([x[0], x[1]], dtype=float)
    return f, grad_f, [g1], [grad_g1]

# 2) Квадратична ціль із лінійними нерівностями: Ax <= b
def quad_linear():
    Q = np.array([[8.0, 3.0],[3.0, 4.0]])
    c = np.array([-2.0, -3.0])
    A = np.array([[1.0, 2.0], [-1.0, 2.0], [0.0, -1.0]])
    b = np.array([2.0, 2.0, 0.0])
    def f(x):
        return 0.5*x@Q@x + c@x
    def grad_f(x):
        return Q@x + c
    g_list = [lambda x, Ai=Ai, bi=bi: Ai@x - bi for Ai, bi in zip(A, b)]
    grad_g_list = [lambda x, Ai=Ai: Ai for Ai in A]
    return f, grad_f, g_list, grad_g_list

def demo():
    # --- приклад 1 ---
    f, gf, gL, gGL = rosenbrock_2d()
    # стартова точка, завідомо допустима: всередині кола
    x0 = np.array([0.0, 0.0])
    res = zoutendijk_feasible_directions(f, gf, gL, gGL, x0, verbose=True)
    print("\n[Розенброк+коло] f* =", res["f"], " x* =", res["x"])

    # --- приклад 2 ---
    f2, gf2, g2, gg2 = quad_linear()
    # знайдемо просту допустиму x0: розв'яжемо A x <= b вручну — беремо x0=(0,0) і перевіримо
    x0 = np.array([0.0, 0.0])
    # Пересунемось усередину, якщо порушено (тут (0,0) уже допустима)
    res2 = zoutendijk_feasible_directions(f2, gf2, g2, gg2, x0, verbose=True)
    print("\n[Quad + лінійні] f* =", res2["f"], " x* =", res2["x"])

if __name__ == "__main__":
    demo()
