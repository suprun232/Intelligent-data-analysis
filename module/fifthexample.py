import numpy as np, os, matplotlib.pyplot as plt
np.random.seed(7)
os.makedirs("figs", exist_ok=True)
def savefig(fig, name):
    path = os.path.join("figs", name); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig); return path


# Дані
t = np.linspace(0, 3, 80)
a0, b0, c0 = 2.0, 0.8, 0.5
y_true = a0*np.exp(b0*t) + c0
y = y_true + 0.1*np.random.randn(len(t))


# Модель і Якобіан
def model(theta, t): a,b,c = theta; return a*np.exp(b*t) + c
def residual(theta): return model(theta, t) - y
def jacobian(theta):
    a,b,c = theta
    e = np.exp(b*t)
    J = np.stack([e, a*t*e, np.ones_like(t)], axis=1)
    return J


# LM
theta = np.array([1.0, 0.1, 0.0])  # старт
lam = 1e-2
loss = []
for it in range(100):
    r = residual(theta)
    J = jacobian(theta)
    g = J.T @ r
    H = J.T @ J
    # Розв'язати (H + lam I) Δ = -g
    delta = np.linalg.solve(H + lam*np.eye(3), -g)
    new_theta = theta + delta
    # Прийняття кроку
    if np.linalg.norm(residual(new_theta)) < np.linalg.norm(r):
        theta = new_theta
        lam *= 1/3
    else:
        lam *= 3
    loss.append(0.5*np.dot(residual(theta), residual(theta)))
    if np.linalg.norm(delta) < 1e-8 or np.linalg.norm(g) < 1e-6: break


fig = plt.figure(figsize=(5,3))
plt.plot(loss); plt.yscale("log")
plt.xlabel("ітерація"); plt.ylabel("loss"); plt.title("LM convergence")
savefig(fig, "lm_convergence.png")
print("theta_est:", theta)
print("[FIG: lm_convergence.png]")
