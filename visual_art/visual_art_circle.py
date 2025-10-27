import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
ax = plt.gca()
ax.set_facecolor("black")
ax.axis("off")

for i in range(len(pt_n)):
    r = pt_n[i] * 400
    theta = phi_n[i] * 2 * np.pi
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    c = plt.cm.plasma(eta_n[i])  # color by pseudorapidity
    a = 0.05 + 0.8 * mass_n[i]   # transparency
    lw = 0.1 + 3 * mass_n[i]     # line width

    # Draw abstract arcs or spirals
    t = np.linspace(0, 4*np.pi, 200)
    x_curve = x + np.sin(t + theta) * (r * 0.1 * eta_n[i])
    y_curve = y + np.cos(t + theta) * (r * 0.1 * eta_n[i])
    ax.plot(x_curve, y_curve, color=c, alpha=a, lw=lw)

plt.tight_layout()
plt.savefig("collision_art.png", dpi=300, facecolor="black")
plt.show()