"""Single source of truth for the Phase-J pilot tensor layout.

PURE NUMPY (no torch): the cache builder runs in the Python-3.14 generation
venv (which has numpy but cannot install torch), while the trainer runs in a
ROCm PyTorch container. Both import this module, so the channel/scalar layout
can never drift between them.

A sample is turned into:
  * target  : (1, R, R)   density in [-1, 1]   (R = OUT_RES, default 64)
  * cond    : (C, R, R)   conditioning rasters (NOT noised at train time)
  * scalars : (S,)        global conditioning vector

Node -> pixel convention (verified empirically against saved densities):
    ix = node % (res + 1);  iy = node // (res + 1);  density[iy, ix].
Ports reference the DESIGN mesh create_mesh(res, res) => (res+1)^2 nodes.
"""

import numpy as np

OUT_RES = 64

# ---- conditioning channels (order is the contract) ----
COND_CHANNELS = [
    "cond_energy",   # 0: uniform-load strain-energy prior (log, per-sample norm)
    "in_blob",       # 1: input-port location (gaussian)
    "in_dir_x",      # 2: input load direction x, localized at the port
    "in_dir_y",      # 3: input load direction y
    "out_blob",      # 4: output-port location
    "out_dir_x",     # 5: desired output motion direction x
    "out_dir_y",     # 6: desired output motion direction y
    "fixed_mask",    # 7: fixed (Dirichlet) support raster
    # 8: allowed design-envelope *coverage* at the model grid.  It is binary
    # at native resolution but fractional after 128→64 mean pooling; never
    # infer it from density or cast it to bool for volume accounting.
    "domain_mask",
    "fixed_x",       # 9: x-constrained support dofs
    "fixed_y",       # 10: y-constrained support dofs
]
COND_DIM = len(COND_CHANNELS)

# ---- scalar (global) conditioning ----
MAGNITUDE_CLASSES = ["force_amp", "transmitting", "displacement_amp",
                     "displacement_reducer"]
TRANSFER_CLASSES = ["inverting", "redirecting", "forwarding"]

SCALAR_NAMES = (
    # Spring stiffnesses are part of the BVP, not labels to be guessed from
    # topology.  log1p keeps their large dynamic range numerically tame.
    ["vf", "ga", "mech_adv", "angle_sin", "angle_cos", "k_in", "k_out", "k_perp"]
    + [f"mag_{c}" for c in MAGNITUDE_CLASSES]
    + [f"trn_{c}" for c in TRANSFER_CLASSES]
)
SCALAR_DIM = len(SCALAR_NAMES)

_BLOB_SIGMA = 1.6  # px, at OUT_RES


def node_to_pixel(node, res, out_res=OUT_RES):
    """Node index on the (res+1)^2 design grid -> (col, row) float in [0, out_res-1]."""
    ix = node % (res + 1)
    iy = node // (res + 1)
    s = (out_res - 1) / res
    return ix * s, iy * s  # (px_col, py_row)


def _splat(px, py, out_res=OUT_RES, sigma=_BLOB_SIGMA):
    """Unit-peak gaussian blob centered at (col=px, row=py)."""
    yy, xx = np.mgrid[0:out_res, 0:out_res].astype(np.float32)
    g = np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2.0 * sigma * sigma))
    return g.astype(np.float32)


def downsample_mean(field, out_res=OUT_RES):
    """Block-mean pool a square field down to out_res. Assumes field is (R, R)
    with R a multiple of out_res (128 -> 64)."""
    r = field.shape[0]
    f = r // out_res
    if f == 1:
        return field.astype(np.float32)
    return field.reshape(out_res, f, out_res, f).mean(axis=(1, 3)).astype(np.float32)


def build_target(density, out_res=OUT_RES):
    """Conservatively rasterize a source density into the v1 model grid.

    This is the frozen training-target contract used by the live v1 cache:
    block means are global-area averages, including zero density outside a
    masked source envelope.  At evaluation, the accompanying ``domain_mask``
    channel is interpreted as fractional area coverage and all 64px methods
    receive the same coverage-weighted volume/envelope projection before
    full-resolution verification.  Do not treat this raster alone as a native
    material-density field; ``reference_native`` remains the full-resolution
    ceiling/check.
    """
    rho = downsample_mean(np.clip(density, 0.0, 1.0), out_res)
    return (2.0 * rho - 1.0)[None].astype(np.float32)


def decode_mask_rle(encoded, res):
    """Decode the compact domain mask stored with v1 samples.

    Historical samples legitimately lack this field; their old contract was a
    full rectangular design region, so preserve that interpretation on read.
    """
    if not encoded:
        return np.ones((res, res), dtype=np.float32)
    shape = tuple(int(v) for v in encoded.get("shape", (res, res)))
    if shape != (res, res):
        raise ValueError(f"domain mask shape {shape} disagrees with resolution {res}")
    total = res * res
    values = []
    state = bool(encoded.get("starts_with", False))
    for run in encoded.get("runs", []):
        values.extend([state] * int(run))
        state = not state
    if len(values) != total:
        raise ValueError("malformed domain_mask_rle")
    return np.asarray(values, dtype=np.float32).reshape(res, res)


def build_cond(meta, cond_energy, out_res=OUT_RES):
    """Assemble the (COND_DIM, out_res, out_res) conditioning stack from the
    sample metadata and its uniform-load strain-energy field."""
    res = int(meta["resolution"])
    mech = meta["mechanism"]
    ch = np.zeros((COND_DIM, out_res, out_res), dtype=np.float32)

    # 0: cond_energy, log + per-sample robust normalize (spatial pattern only;
    # absolute magnitude is carried by scalars). Robust to outliers via 99pct.
    ce = downsample_mean(np.asarray(cond_energy, dtype=np.float32), out_res)
    ce = np.log1p(np.clip(ce, 0.0, None))
    hi = np.percentile(ce, 99.0)
    ch[0] = ce / hi if hi > 1e-9 else ce

    # 1-3: input port + load direction
    ipx, ipy = node_to_pixel(int(mech["input_node"]), res, out_res)
    iblob = _splat(ipx, ipy, out_res)
    idir = np.asarray(mech["input_direction"], dtype=np.float32)
    idir = idir / (np.linalg.norm(idir) + 1e-9)
    ch[1] = iblob
    ch[2] = iblob * idir[0]
    ch[3] = iblob * idir[1]

    # 4-6: output port + desired motion direction
    opx, opy = node_to_pixel(int(mech["output_node"]), res, out_res)
    oblob = _splat(opx, opy, out_res)
    odir = np.asarray(mech["output_direction"], dtype=np.float32)
    odir = odir / (np.linalg.norm(odir) + 1e-9)
    ch[4] = oblob
    ch[5] = oblob * odir[0]
    ch[6] = oblob * odir[1]

    # 7/9/10: supports, retaining per-DOF direction instead of collapsing a
    # roller into a full clamp.  Channel 7 remains for v0 checkpoint readers.
    fixed = np.zeros((out_res, out_res), dtype=np.float32)
    fixed_x = np.zeros_like(fixed)
    fixed_y = np.zeros_like(fixed)
    for bc in meta.get("boundary_conditions", []):
        dirs = set(int(d) for d in bc.get("directions", (0, 1)))
        for node in bc.get("nodes", []):
            px, py = node_to_pixel(int(node), res, out_res)
            r = int(round(py)); c = int(round(px))
            if 0 <= r < out_res and 0 <= c < out_res:
                fixed[r, c] = 1.0
                if 0 in dirs:
                    fixed_x[r, c] = 1.0
                if 1 in dirs:
                    fixed_y[r, c] = 1.0
    ch[7] = fixed
    # At OUT_RES < res this is a fractional native-area coverage field, not a
    # Boolean occupancy mask.  Sampling/evaluation must retain the weights.
    domain = decode_mask_rle(meta.get("domain_mask_rle"), res)
    ch[8] = downsample_mean(domain, out_res)
    ch[9] = fixed_x
    ch[10] = fixed_y
    return ch


def build_scalars(meta):
    """Global conditioning vector (SCALAR_DIM,). Robust to missing motion label
    (older samples): falls back to geometry-derived angle and zeroed one-hots."""
    s = np.zeros(SCALAR_DIM, dtype=np.float32)
    idx = {n: i for i, n in enumerate(SCALAR_NAMES)}

    s[idx["vf"]] = float(meta.get("volume_fraction_target", 0.0))

    val = meta.get("validation", {})
    motion = val.get("motion", {}) or {}
    quality = val.get("quality", {}) or {}

    ga = motion.get("ga_signed", quality.get("ga", 0.0))
    s[idx["ga"]] = float(np.tanh(float(ga) / 2.0))            # bounded
    s[idx["mech_adv"]] = float(np.clip(motion.get("mech_advantage", 0.0), 0, 4) / 4.0)
    mech = meta.get("mechanism", {})
    for name in ("k_in", "k_out", "k_perp"):
        s[idx[name]] = float(np.log1p(max(0.0, float(mech.get(name, 0.0)))) / np.log1p(100.0))

    if "transfer_angle_deg" in motion:
        ang = np.deg2rad(float(motion["transfer_angle_deg"]))
    else:
        # derive from stored port directions
        a = np.asarray(meta["mechanism"]["input_direction"], float)
        b = np.asarray(meta["mechanism"]["output_direction"], float)
        a /= np.linalg.norm(a) + 1e-9
        b /= np.linalg.norm(b) + 1e-9
        ang = np.arccos(np.clip(a @ b, -1, 1))
    s[idx["angle_sin"]] = float(np.sin(ang))
    s[idx["angle_cos"]] = float(np.cos(ang))

    mc = motion.get("magnitude_class")
    if mc in MAGNITUDE_CLASSES:
        s[idx[f"mag_{mc}"]] = 1.0
    tc = motion.get("transfer_class")
    if tc in TRANSFER_CLASSES:
        s[idx[f"trn_{tc}"]] = 1.0
    return s
