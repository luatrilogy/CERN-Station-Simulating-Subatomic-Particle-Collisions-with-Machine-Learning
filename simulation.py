import numpy as np, time, pandas as pd
import tensorflow as tf 
from keras.models import load_model

# Load for inference only
generator = load_model("generator_model.h5", compile=False)

def get_latent_dim(model):
    ishape = model.input_shape
    if isinstance(ishape, (list, tuple)) and isinstance(ishape[0], (list, tuple)):
        ishape = ishape[0]
    return int(ishape[-1])

def generate_events_auto(gen, n_events=20000, batch_size=2048, seed=42, outfile="gan_20k.csv"):
    z_dim = get_latent_dim(gen)
    rng = np.random.default_rng(seed)
    outputs = []
    t0 = time.perf_counter()

    remaining = n_events
    while remaining > 0:
        b = min(batch_size, remaining)
        z = rng.standard_normal(size=(b, z_dim)).astype("float32")
        x_fake = gen(z, training=False).numpy()  # inference path
        outputs.append(x_fake)
        remaining -= b

    elapsed = time.perf_counter() - t0
    samples = np.vstack(outputs)[:n_events]
    pd.DataFrame(samples).to_csv(outfile, index=False)
    print(f"Generated {n_events} events in {elapsed:.4f}s "
          f"({elapsed/n_events*1000:.3f} ms/event). Saved -> {outfile}")
    return samples, elapsed

# Run
events, elapsed = generate_events_auto(generator, n_events=100000, batch_size=2048, outfile="gan_100k.csv")
