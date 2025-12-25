import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def fmt_time(us: float) -> str:
    if us >= 1000:
        return f"{us/1000:.1f} ms"
    if us >= 10:
        return f"{us:.0f} µs"
    return f"{us:.1f} µs"


df = pd.read_csv("bench_results.csv")

# short labels
label_map = {
    "GAN (generator_model.h5)": "GAN",
    "PYTHIA8": "PYTHIA8",
}
df["label"] = df["method"].map(label_map).fillna(df["method"])

# keep just the two methods you care about
df = df[df["label"].isin(["GAN", "PYTHIA8"])].copy()
df = df.set_index("label").loc[["GAN", "PYTHIA8"]].reset_index()

vals = df["us_per_event"].values
errs = df["time_std_s"].values / df["n_events"].values * 1e6  # std in µs/event

fig, axes = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"wspace": 0.35})

# --- left: log scale ---
axes[0].bar(df["label"], vals, yerr=errs)
axes[0].set_yscale("log")
axes[0].set_title("Time per event (log scale)")
axes[0].set_ylabel("Time per event")

# annotate values
for i, v in enumerate(vals):
    axes[0].text(i, v, fmt_time(v), ha="center", va="bottom")

speedup = vals[1] / vals[0]
axes[0].text(
    0.5, -0.1,                       # y < 0 puts it below the axes
    f"Speedup ≈ {speedup:.0f}×",
    transform=axes[0].transAxes,
    ha="center", va="top",
    clip_on=False,
    bbox=dict(facecolor="white", edgecolor="none", pad=2)
)


# --- right: linear zoom on GAN ---
axes[1].bar(["GAN"], [vals[0]], yerr=[errs[0]])
axes[1].set_title("GAN zoom (linear)")
axes[1].set_ylabel("µs/event")

plt.suptitle(f"Event generation speed on same machine (N={int(df['n_events'].iloc[0]):,})")
plt.tight_layout(rect=[0, 0.08, 1, 1])   # leave ~8% space at bottom
plt.savefig("speed_comparison_twopanel.png", dpi=300)
plt.show()
