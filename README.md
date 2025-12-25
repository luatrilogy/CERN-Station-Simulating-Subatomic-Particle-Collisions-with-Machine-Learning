# CERN Station
# Generative Visualization of Particle Collisions Using Physics-Informed Mappings
paper: https://drive.google.com/file/d/1oXyoA6GCsiiKh0N340yiJsB-nDRPDa_W/view?usp=sharing
training data: https://www.kaggle.com/datasets/fedesoriano/cern-electron-collision-data?resource=download 

# OVERVIEW
CERN Station is a generative system that transforms particle-collision data into visual art while preserving interpretable mappings to underlying physical quantities. Rather than simulating a detector or producing literal event displays, the project explores how high-energy physics data can be translated into structured, legible visual systems that balance scientific meaning with artistic expression.

The system was designed as both an educational tool and a creative medium, allowing viewers to observe patterns across many collision events simultaneously.

TO GENERATE COLLISIONS
python app.py

TO GENERATE VISUALIZATION
python visual_art_gorge.py `
  --csv "A:\My Stuff\Projects\Python Stuff\Simulating Particals\gen\gan_100k.csv" `
  --out "collision_art_dense_distorted3.png" `
  --sym 6 --seed 11 --cmap magma `
  --distort_strength 0.006 --ripple_amp 0.003 --chaos_scale 0.002 `
  --progress

# MOTIVATION
Traditional particle-collision visualizations focus on individual events or detector realism, which can make it difficult for students to see global structure or distributional behavior. I wanted to explore an alternative question:

    How can collision data be represented so that statistical structure and physical intuition emerge visually, without requiring domain-specific detector knowledge?

CERN Station approaches this by mapping core kinematic quantities directly into geometric and stylistic visual parameters.

# Data & Inputs
The system operates on per-particle or per-event data with fields such as:
    Transverse momentum (pT)
    Azimuthal angle (φ)
    Pseudorapidity (η)
    Mass (m)
Data may come from real detector datasets or synthetically generated events (e.g., via machine-learning models), provided the distributions resemble physical collisions.

# Physics-to-Visual Mapping
Each visual element corresponds to a specific physical quantity:
    Azimuth (φ) → angular position around the center
    Transverse momentum (pT) → radial distance from the center
    Pseudorapidity (η) → color (via continuous colormap)
    Mass (m) → line width and opacity
A small φ-dependent radial term introduces a controlled swirl that enhances legibility without obscuring structure.

# Styling Layers
After physics-based placement, optional stylistic layers are applied:
    Rotational symmetry tiling (repeating events around the circle)
    Distortion, ripple, or chaos-based warping to introduce organic texture
These layers are applied after feature mapping, preserving a clear boundary between encoded data and artistic interpretation.

# Rendering Pipeline
At a high level, the pipeline follows:
    1. Load or generate collision event data
    2. Normalize physical quantities
    3. Map physics variables to geometric and visual parameters
    4. Apply symmetry tiling
    5. Apply optional distortion / warp layers
    6. Render composite image
The system is designed to scale to tens of thousands of events, allowing aggregate structure to emerge.

# Educational & Artistic Use
CERN Station is used in two primary ways:
    Education: helping students visually explore collision behavior without requiring detector expertise
    Art: producing generative works that remain grounded in real physical structure
This dual purpose is intentional; accessibility and rigor are treated as complementary goals.