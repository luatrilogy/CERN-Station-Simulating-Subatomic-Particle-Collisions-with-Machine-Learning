import os, time, platform, subprocess, math
import numpy as np
import pandas as pd

# ---------- Optional progress bar ----------
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def pbar(iterable, total=None, desc=None, leave=True):
    """tqdm if available, otherwise just return the iterable."""
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=leave)

# ---------- GAN runner ----------
import tensorflow as tf
from keras.models import load_model

GEN_PATH = os.getenv(
    "GEN_PATH",
    "gen/generator_model.h5"  # recommended: relative path from project root
)

def get_latent_dim(model):
    ishape = model.input_shape
    if isinstance(ishape, (list, tuple)) and isinstance(ishape[0], (list, tuple)):
        ishape = ishape[0]
    return int(ishape[-1])

def run_gan(generator_path, n_events=100_000, batch_size=2048, seed=42, warmup_batches=5):
    gen = load_model(generator_path, compile=False)
    z_dim = get_latent_dim(gen)
    rng = np.random.default_rng(seed)

    # warmup (important for GPU + TF graph caches)
    for _ in range(warmup_batches):
        z = rng.standard_normal(size=(batch_size, z_dim)).astype("float32")
        _ = gen(z, training=False).numpy()  # forces sync for realistic timing

    # timed region
    n_batches = math.ceil(n_events / batch_size)
    t0 = time.perf_counter()

    # progress bar over batches (one bar per run)
    for i in pbar(range(n_batches), total=n_batches, desc=f"GAN batches ({batch_size})", leave=False):
        b = batch_size if i < n_batches - 1 else (n_events - batch_size * (n_batches - 1))
        z = rng.standard_normal(size=(b, z_dim)).astype("float32")
        _ = gen(z, training=False).numpy()

    elapsed = time.perf_counter() - t0
    return elapsed

# ---------- Baseline runner template ----------
def run_external_command(cmd_list, heartbeat_sec=2, timeout_sec=None):
    """
    Runs a command while printing a heartbeat so it doesn't look stuck.
    If timeout_sec is set, kills the process if it exceeds that time.
    """
    t0 = time.perf_counter()
    p = subprocess.Popen(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    last = time.perf_counter()
    while True:
        ret = p.poll()
        now = time.perf_counter()

        # heartbeat
        if now - last >= heartbeat_sec:
            elapsed = now - t0
            print(f"[running] {' '.join(cmd_list)}  elapsed={elapsed:.1f}s")
            last = now

        # optional timeout
        if timeout_sec is not None and (now - t0) > timeout_sec:
            p.kill()
            out, err = p.communicate()
            raise TimeoutError(
                f"Command timed out after {timeout_sec}s:\n"
                f"CMD: {' '.join(cmd_list)}\nSTDOUT:\n{out}\nSTDERR:\n{err}"
            )

        if ret is not None:
            out, err = p.communicate()
            elapsed = time.perf_counter() - t0
            if ret != 0:
                raise RuntimeError(
                    f"Command failed ({ret}).\nCMD: {' '.join(cmd_list)}\nSTDOUT:\n{out}\nSTDERR:\n{err}"
                )
            return elapsed

# ---------- Machine info ----------
def machine_info():
    info = {
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor(),
        "tf": tf.__version__,
    }
    gpus = tf.config.list_physical_devices("GPU")
    info["gpu"] = gpus[0].name if gpus else "none"
    return info

# ---------- Benchmark loop ----------
def bench(method_name, fn, n_events, repeats=5, **kwargs):
    times = []

    # progress bar over repeats
    for r in pbar(range(repeats), total=repeats, desc=f"{method_name} repeats", leave=True):
        t = fn(n_events=n_events, **kwargs)
        times.append(t)

    times = np.array(times)
    return {
        "method": method_name,
        "n_events": n_events,
        "repeats": repeats,
        "time_mean_s": float(times.mean()),
        "time_std_s": float(times.std(ddof=1)) if repeats > 1 else 0.0,
        "events_per_s": float(n_events / times.mean()),
        "us_per_event": float(times.mean() / n_events * 1e6),
        **machine_info()
    }

if __name__ == "__main__":
    N = 50000  # <= 70000 | change it in pythia_driver.cpp too
    rows = []

    # --- GAN ---
    rows.append(bench(
        "GAN (generator_model.h5)",
        lambda n_events, batch_size=2048, seed=42: run_gan(
            GEN_PATH,
            n_events=n_events,
            batch_size=batch_size,
            seed=seed
        ),
        n_events=N,
        repeats=5,
        batch_size=2048,
        seed=42
    ))

    rows.append(bench(
        "PYTHIA8",
        lambda n_events: run_external_command(
            ["./pythia_driver", "--nev", str(n_events), "--seed", "12345"],
            heartbeat_sec=2,
            timeout_sec=300  # 5 minutes; adjust if needed
        ),
        n_events=N,
        repeats=5
    ))

    out = pd.DataFrame(rows)
    
    print("CWD:", os.getcwd())
    print("Writing CSV to:", os.path.abspath("bench_results.csv"))
    print("Rows:", len(rows), "Methods:", [r.get("method") for r in rows])

    out.to_csv("bench_results.csv", index=False)
    print("Wrote CSV OK")
    print(out[["method", "time_mean_s", "time_std_s", "events_per_s", "us_per_event", "gpu", "cpu"]])
