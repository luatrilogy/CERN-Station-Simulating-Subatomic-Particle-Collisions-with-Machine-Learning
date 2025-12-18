import matplotlib.pyplot as plt
import matplotlib.image as mpimg

IN_IMG  = "collision_art_dense_distorted.png"
OUT_IMG = "collision_art_dense_distorted_annotated.png"

img = mpimg.imread(IN_IMG)

fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
ax.imshow(img)
ax.axis("off")

# Helper for consistent arrow labels
def label(text, xy, xytext):
    ax.annotate(
        text,
        xy=xy, xycoords="axes fraction",
        xytext=xytext, textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", lw=1.2),
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", alpha=0.85),
    )

# These labels correspond directly to your mapping logic:
label("φ (azimuth) → angle around center",       xy=(0.58, 0.55), xytext=(0.78, 0.18))
label("pT → radial distance / scale",            xy=(0.50, 0.82), xytext=(0.08, 0.18))
label("η (pseudorapidity) → color (colormap)",   xy=(0.78, 0.62), xytext=(0.72, 0.88))
label("m (mass) → line width (and opacity)",     xy=(0.32, 0.45), xytext=(0.06, 0.78))
label("symmetry tiling: repeated rotations",     xy=(0.50, 0.50), xytext=(0.35, 0.05))
label("distortion/ripple/chaos warp = style layer", xy=(0.62, 0.72), xytext=(0.68, 0.70))

plt.tight_layout(pad=0)
plt.savefig(OUT_IMG, dpi=300, bbox_inches="tight", pad_inches=0)
plt.close(fig)
print("Saved:", OUT_IMG)