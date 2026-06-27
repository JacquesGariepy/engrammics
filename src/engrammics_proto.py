"""
Distributed engrammics -- minimal, self-contained prototype (NumPy).

Central object: a fast-weight matrix F (the engram) written by the delta rule,
read associatively, transferred peer-to-peer as a low-rank delta, and forgotten
by projection.

Demonstrates four falsifiable claims:
  C1  Persistence       : F survives a save / reload.
  C2  Gradient-free transfer : a skill learned by A transfers to B via a rank-r
                          delta, integrated without any gradient step, with a
                          fidelity that grows with r.
  C3  Targeted forgetting : removing a skill by projection does not degrade the
                          others (right to be forgotten = linear algebra).
  C4  Governance        : the receiver admits an engram only within the
                          consented subspace; the rest is cancelled.

The "conatus" (slow controller) is here a linear projection trained to inscribe
non-interfering engrams (~orthonormal keys). No GPU required.
To scale to a real LM, this toy memory is replaced by the recurrent state of a
DeltaNet / Gated DeltaNet (flash-linear-attention): same F, same delta rule.
"""

import numpy as np

SEED = 0
rng = np.random.default_rng(SEED)

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
D_IN = 48     # dimension of a cue / of a raw response
D_K = 64      # dimension of the keys (size of F's addressing space)
D_V = D_IN    # values = responses directly (value encoder = identity)
N_PAIRS = 8   # associations per skill for the C1-C4 demonstrations
N_MIN, N_MAX = 8, 32   # capacity range seen during conatus training
BETA = 1.0    # delta-rule write rate (exact correction)


# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------
def l2norm(X, axis=-1, eps=1e-9):
    return X / (np.linalg.norm(X, axis=axis, keepdims=True) + eps)


def sample_skill(rng, n=N_PAIRS, d_in=D_IN, d_v=D_V):
    """A 'skill' = an associative dictionary cue -> response."""
    cues = rng.standard_normal((n, d_in))
    responses = l2norm(rng.standard_normal((n, d_v)))
    return cues, responses


# ----------------------------------------------------------------------------
# Conatus (slow controller): cue -> key. Value = response (identity).
# ----------------------------------------------------------------------------
def init_conatus(d_k=D_K, d_in=D_IN, rng=rng):
    return rng.standard_normal((d_k, d_in)) / np.sqrt(d_in)


def encode_keys(Wk, cues):
    """cues (N,d_in) -> normalized keys (N,d_k)."""
    return l2norm(cues @ Wk.T)


# ----------------------------------------------------------------------------
# Fast-weight memory F  (the 'read' / 'inscribe')
# ----------------------------------------------------------------------------
def delta_write(S, K, V, beta=BETA):
    """Inscribe: sequential write by the delta rule (error correction)."""
    S = S.copy()
    for i in range(K.shape[0]):
        pred = K[i] @ S                      # current prediction (d_v,)
        S = S + beta * np.outer(K[i], V[i] - pred)
    return S


def read(S, Q):
    """Read: associative read o = q . F."""
    return Q @ S


def recall_metrics(O, V_pool):
    """Top-1 (cosine, nearest) AND mean signal (cosine to the true response).
    The signal is continuous: it collapses cleanly when the trace is erased,
    whereas top-1 on noise remains an argmax artifact."""
    On, Vn = l2norm(O), l2norm(V_pool)
    sims = On @ Vn.T
    acc = float((sims.argmax(axis=1) == np.arange(O.shape[0])).mean())
    signal = float(np.mean(np.sum(On * Vn, axis=1)))
    return acc, signal


def build_engram(Wk, cues, responses, beta=BETA):
    """Builds the engram F of a skill from a zero state."""
    S0 = np.zeros((D_K, D_V))
    K = encode_keys(Wk, cues)
    return delta_write(S0, K, responses, beta), K


# ----------------------------------------------------------------------------
# The novel operations: transfer / forget / govern
# ----------------------------------------------------------------------------
def low_rank(S, r):
    """Transfer: truncate the engram to rank r (compact delta)."""
    U, s, Vt = np.linalg.svd(S, full_matrices=False)
    return (U[:, :r] * s[:r]) @ Vt[:r]


def forget_projection(S, K_x):
    """Forget: remove from F the component read by skill X's keys."""
    P = K_x.T @ np.linalg.pinv(K_x @ K_x.T) @ K_x   # projector onto span(keys X)
    return S - P @ S


def orthobasis(K):
    """Orthonormal basis (d_k x r) of the space spanned by the keys (rows of K)."""
    _, s, Vt = np.linalg.svd(K, full_matrices=False)
    r = int((s > 1e-8).sum())
    return Vt[:r].T


def govern_projection(dS, U):
    """Governance: admit the engram only within the consented subspace U."""
    P = U @ U.T
    return P @ dS


# ----------------------------------------------------------------------------
# Conatus training: inscribe non-interfering engrams
#   loss = || K K^T - I ||_F^2  (orthonormal keys -> low crosstalk)
# ----------------------------------------------------------------------------
def ortho_loss_and_grad(Wk, cues):
    Z = cues @ Wk.T                          # (N, d_k)
    norm = np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9
    K = Z / norm                             # normalized keys
    N = K.shape[0]
    G = K @ K.T - np.eye(N)
    loss = float((G * G).sum())
    dK = 4.0 * G @ K                         # dL/dK
    # backprop through the (row-wise) L2 normalization
    dot = np.sum(dK * K, axis=1, keepdims=True)
    dZ = (dK - K * dot) / norm
    dWk = dZ.T @ cues                        # dL/dWk
    return loss, dWk


def train_conatus(Wk, steps=800, batch=16, lr=0.08, rng=rng):
    """The conatus learns to inscribe non-interfering engrams, over a range of
    sizes N (variable capacity)."""
    for _ in range(steps):
        g = np.zeros_like(Wk)
        for _ in range(batch):
            n = int(rng.integers(N_MIN, N_MAX + 1))
            cues, _ = sample_skill(rng, n=n)
            _, gi = ortho_loss_and_grad(Wk, cues)
            g += gi
        Wk = Wk - lr * (g / batch)
    return Wk


def mean_recall(Wk, n_pairs=N_PAIRS, n_skills=200, rng=rng):
    accs, sigs = [], []
    for _ in range(n_skills):
        cues, resp = sample_skill(rng, n=n_pairs)
        S, _ = build_engram(Wk, cues, resp)
        a, s = recall_metrics(read(S, encode_keys(Wk, cues)), resp)
        accs.append(a); sigs.append(s)
    return float(np.mean(accs)), float(np.mean(sigs))


# ----------------------------------------------------------------------------
# Gradient check (guard against a wrong derivation)
# ----------------------------------------------------------------------------
def gradient_check():
    Wk = init_conatus(d_k=8, d_in=6, rng=np.random.default_rng(1))
    cues, _ = sample_skill(np.random.default_rng(2), n=4, d_in=6, d_v=6)
    _, g = ortho_loss_and_grad(Wk, cues)
    eps = 1e-5
    num = np.zeros_like(Wk)
    for a in range(Wk.shape[0]):
        for b in range(Wk.shape[1]):
            Wp = Wk.copy(); Wp[a, b] += eps
            Wm = Wk.copy(); Wm[a, b] -= eps
            lp, _ = ortho_loss_and_grad(Wp, cues)
            lm, _ = ortho_loss_and_grad(Wm, cues)
            num[a, b] = (lp - lm) / (2 * eps)
    rel = np.linalg.norm(g - num) / (np.linalg.norm(g) + np.linalg.norm(num) + 1e-12)
    return rel


# ============================================================================
# Statistics and alignment utilities
# ============================================================================
def agg(vals):
    a = np.array(vals, dtype=float)
    return float(a.mean()), float(a.std())


def pm(m, s, pct=True):
    return f"{m*100:5.1f} +/- {s*100:4.1f}" if pct else f"{m:5.2f} +/- {s:4.2f}"


def align_keyspace(Wk_src, Wk_dst, anchors):
    """Learns W such that (dst keys) @ W approximates (src keys), from a set of
    shared anchors. Lets a dst agent address a src agent's engram despite a
    different key space (Wk_src != Wk_dst)."""
    K_src = encode_keys(Wk_src, anchors)
    K_dst = encode_keys(Wk_dst, anchors)
    W, *_ = np.linalg.lstsq(K_dst, K_src, rcond=None)   # K_dst @ W ~= K_src
    return W


# ============================================================================
# One experiment = one seed. We then aggregate mean +/- std.
# ============================================================================
def run_core_trial(Wk, seed):
    rt = np.random.default_rng(seed)
    cues_X, resp_X = sample_skill(rt)
    cues_Y, resp_Y = sample_skill(rt)
    S_A, K_X = build_engram(Wk, cues_X, resp_X)
    S_Y, K_Y = build_engram(Wk, cues_Y, resp_Y)
    o = {}
    for r in (1, 2, 4, 8):
        o[f"tx{r}"] = recall_metrics(read(S_Y + low_rank(S_A, r), K_X), resp_X)[0]
    o["ty4"] = recall_metrics(read(S_Y + low_rank(S_A, 4), K_Y), resp_Y)[0]
    S_f = forget_projection(S_A + S_Y, K_X)
    o["forget_sigX"] = recall_metrics(read(S_f, K_X), resp_X)[1]
    o["forget_retY"] = recall_metrics(read(S_f, K_Y), resp_Y)[0]
    dS = low_rank(S_A, N_PAIRS)
    Ua = orthobasis(K_X)
    R = rt.standard_normal((D_K, K_X.shape[0])); R = R - Ua @ (Ua.T @ R)
    Ud = orthobasis(R.T)
    o["gov_admit"] = recall_metrics(read(S_Y + govern_projection(dS, Ua), K_X), resp_X)[0]
    o["gov_deny"] = recall_metrics(read(S_Y + govern_projection(dS, Ud), K_X), resp_X)[0]
    return o


def run_hetero_trial(seed, n_anchor=128):
    """A and B have different key spaces. We compare naive transfer vs transfer
    with an alignment learned on shared anchors."""
    rt = np.random.default_rng(seed)
    Wk_A = init_conatus(rng=rt)
    Wk_B = init_conatus(rng=rt)
    cues_X, resp_X = sample_skill(rt)
    S_A, K_XA = build_engram(Wk_A, cues_X, resp_X)
    K_XB = encode_keys(Wk_B, cues_X)
    homo = recall_metrics(read(S_A, K_XA), resp_X)[0]            # same Wk: upper bound
    naive = recall_metrics(read(S_A, K_XB), resp_X)[0]          # different Wk, no alignment
    W = align_keyspace(Wk_A, Wk_B, rt.standard_normal((n_anchor, D_IN)))
    aligned = recall_metrics(read(S_A, K_XB @ W), resp_X)[0]     # with alignment
    return homo, naive, aligned


def alpha_sweep(Wk, seeds, r=4):
    alphas = [0.25, 0.5, 1.0, 1.5, 2.0]
    out = {a: ([], []) for a in alphas}
    for sd in seeds:
        rt = np.random.default_rng(sd)
        cues_X, resp_X = sample_skill(rt); cues_Y, resp_Y = sample_skill(rt)
        S_A, K_X = build_engram(Wk, cues_X, resp_X)
        S_Y, K_Y = build_engram(Wk, cues_Y, resp_Y)
        dS = low_rank(S_A, r)
        for a in alphas:
            S_B = S_Y + a * dS
            out[a][0].append(recall_metrics(read(S_B, K_X), resp_X)[0])
            out[a][1].append(recall_metrics(read(S_B, K_Y), resp_Y)[0])
    return alphas, out


def overlap_sweep(Wk, seeds):
    """Controlled overlap of the key subspaces between X and Y, then forgetting
    of X: shows the graceful degradation of Y's retention."""
    overlaps = [0.0, 0.25, 0.5, 0.75, 1.0]
    out = {ov: [] for ov in overlaps}
    for sd in seeds:
        rt = np.random.default_rng(sd)
        cues_X, resp_X = sample_skill(rt)
        S_A, K_X = build_engram(Wk, cues_X, resp_X)
        P = cues_X.T @ np.linalg.pinv(cues_X @ cues_X.T) @ cues_X   # proj onto span(cues X)
        for ov in overlaps:
            fresh = rt.standard_normal((N_PAIRS, D_IN))
            cues_Y = (1 - ov) * fresh + ov * (fresh @ P)           # Y more or less inside X
            resp_Y = l2norm(rt.standard_normal((N_PAIRS, D_V)))
            S_Y, K_Y = build_engram(Wk, cues_Y, resp_Y)
            S_f = forget_projection(S_A + S_Y, K_X)
            out[ov].append(recall_metrics(read(S_f, K_Y), resp_Y)[0])
    return overlaps, out


def noise_sweep(Wk, seeds, r=8):
    """Gaussian corruption of the transmitted engram (noisy channel / mild
    poisoning): sensitivity of recall to noise."""
    sigmas = [0.0, 0.05, 0.1, 0.2, 0.4]
    out = {s: [] for s in sigmas}
    for sd in seeds:
        rt = np.random.default_rng(sd)
        cues_X, resp_X = sample_skill(rt)
        S_A, K_X = build_engram(Wk, cues_X, resp_X)
        dS = low_rank(S_A, r); scale = dS.std()
        for sg in sigmas:
            noisy = dS + sg * scale * rt.standard_normal(dS.shape)
            out[sg].append(recall_metrics(read(noisy, K_X), resp_X)[0])
    return sigmas, out


# ============================================================================
# Driver
# ============================================================================
def main():
    N_SEEDS = 20
    seeds = list(range(N_SEEDS))
    print("=" * 70)
    print(f"DISTRIBUTED ENGRAMMICS -- prototype  ({N_SEEDS} seeds, mean +/- std)")
    print("=" * 70)

    rel = gradient_check()
    print(f"\n[gradient check] relative error vs finite differences: {rel:.2e}"
          f"  ({'OK' if rel < 1e-5 else 'CHECK'})")

    # --- Conatus: capacity --------------------------------------------------
    Wk0 = init_conatus()
    Wk = train_conatus(Wk0.copy())
    print("\n--- Conatus: learning to inscribe (recall vs load) --------------")
    print(f"{'N assoc.':>9} | {'init':>16} | {'trained':>16}")
    for n in [8, 16, 24, 32]:
        a0, _ = mean_recall(Wk0, n_pairs=n, n_skills=120, rng=np.random.default_rng(100 + n))
        a1, _ = mean_recall(Wk, n_pairs=n, n_skills=120, rng=np.random.default_rng(100 + n))
        print(f"{n:>9} | {a0*100:13.1f}%  | {a1*100:13.1f}%")

    # --- C1: persistence (deterministic) ------------------------------------
    cx, rx = sample_skill(np.random.default_rng(0))
    S0, KX0 = build_engram(Wk, cx, rx)
    np.save("/tmp/F_state.npy", S0); S1 = np.load("/tmp/F_state.npy")
    print("\n--- C1  Persistence ---------------------------------------------")
    print(f"recall before/after save-load: {recall_metrics(read(S0,KX0),rx)[0]*100:.0f}%"
          f" / {recall_metrics(read(S1,KX0),rx)[0]*100:.0f}%"
          f"   (max diff on F: {np.abs(S0-S1).max():.1e})")

    # --- Multi-seed core: C2, C3, C4 ----------------------------------------
    trials = [run_core_trial(Wk, s) for s in seeds]
    st = {k: agg([t[k] for t in trials]) for k in trials[0]}

    print("\n--- C2  Gradient-free transfer (controlled regime) --------------")
    print("A holds X; B already holds Y; B integrates a rank-r delta.")
    print(f"{'rank r':>7} | {'cost (floats)':>13} | {'recall X in B':>18}")
    for r in (1, 2, 4, 8):
        print(f"{r:>7} | {r*(D_K+D_V):>13} | {pm(*st[f'tx{r}']):>18}")
    print(f"Y preserved (at rank 4): {pm(*st['ty4'])} %")
    print(f"Raw replay reference: {N_PAIRS*(D_IN+D_V)} floats. "
          f"Gain only if the skill is compressible (here rank <= 4).")

    print("\n--- C3  Targeted forgetting (fast substrate) --------------------")
    print(f"signal of X after projection: {pm(*st['forget_sigX'], pct=False)} "
          f"(collapses toward 0)")
    print(f"retention of Y (~disjoint subspaces): {pm(*st['forget_retY'])} %")

    print("\n--- C4  Governance (consent = admitted subspace) ----------------")
    print(f"engram inside the admitted subspace: {pm(*st['gov_admit'])} %")
    print(f"engram outside the admitted subspace: {pm(*st['gov_deny'])} %")

    # --- Hetero: agents with different key spaces ----------------------------
    hs = [run_hetero_trial(s) for s in seeds]
    homo = agg([h[0] for h in hs]); naive = agg([h[1] for h in hs])
    aligned = agg([h[2] for h in hs])
    print("\n--- Heterogeneous agents  Wk_A != Wk_B --------------------------")
    print(f"same key space (upper bound)        : {pm(*homo)} %")
    print(f"naive transfer (no alignment)       : {pm(*naive)} %   <- the delta is meaningless")
    print(f"with learned alignment (128 anchors): {pm(*aligned)} %")

    # --- Sensitivities ------------------------------------------------------
    alphas, asweep = alpha_sweep(Wk, seeds)
    print("\n--- Sensitivity to the integration factor alpha (rank 4) -------")
    for a in alphas:
        mx, sx = agg(asweep[a][0]); my, sy = agg(asweep[a][1])
        print(f"alpha={a:>4} : X in B {pm(mx,sx)} %   |  Y preserved {pm(my,sy)} %")

    overlaps, osweep = overlap_sweep(Wk, seeds)
    print("\n--- Forgetting vs subspace overlap (retention of Y) -------------")
    for ov in overlaps:
        print(f"overlap={ov:>4} : retention Y {pm(*agg(osweep[ov]))} %")

    sigmas, nsweep = noise_sweep(Wk, seeds)
    print("\n--- Robustness to noise on the transmitted engram (rank 8) ------")
    for sg in sigmas:
        print(f"sigma={sg:>4} : recall X {pm(*agg(nsweep[sg]))} %")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
