"""
Engrammique distribuee -- prototype minimal et auto-suffisant (NumPy).

Objet central : une matrice de poids rapides F (l'engramme) ecrite par la
delta rule, lue associativement, transferee en pair-a-pair sous forme d'un
delta de rang faible, et oubliee par projection.

Demontre quatre revendications falsifiables :
  C1  Persistance      : F survit a une sauvegarde / rechargement.
  C2  Transfert sans gradient : une competence apprise par A se transmet a B
                          via un delta de rang r, integre sans aucun pas de
                          gradient, avec une fidelite qui croit avec r.
  C3  Oubli cible      : retirer une competence par projection ne degrade pas
                          les autres (droit a l'oubli = algebre lineaire).
  C4  Gouvernance      : le receveur n'admet un engramme que dans le
                          sous-espace consenti ; le reste est annule.

Le "conatus" (controleur lent) est ici une projection lineaire entrainee a
inscrire des engrammes non interferents (cles ~orthonormales). Aucun GPU requis.
Pour passer a l'echelle d'un vrai LM, on remplace cette memoire jouet par l'etat
recurrent d'un DeltaNet / Gated DeltaNet (flash-linear-attention) : meme F,
meme delta rule.
"""

import numpy as np

SEED = 0
rng = np.random.default_rng(SEED)

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
D_IN = 48     # dimension d'un indice (cue) / d'une reponse brute
D_K = 64      # dimension des cles (taille de l'espace d'adressage de F)
D_V = D_IN    # valeurs = reponses directement (encodeur de valeur = identite)
N_PAIRS = 8   # associations par competence pour les demonstrations C1-C4
N_MIN, N_MAX = 8, 32   # plage de capacite vue a l'entrainement du conatus
BETA = 1.0    # taux d'ecriture de la delta rule (correction exacte)


# ----------------------------------------------------------------------------
# Utilitaires
# ----------------------------------------------------------------------------
def l2norm(X, axis=-1, eps=1e-9):
    return X / (np.linalg.norm(X, axis=axis, keepdims=True) + eps)


def sample_skill(rng, n=N_PAIRS, d_in=D_IN, d_v=D_V):
    """Une 'competence' = un dictionnaire associatif cue -> reponse."""
    cues = rng.standard_normal((n, d_in))
    responses = l2norm(rng.standard_normal((n, d_v)))
    return cues, responses


# ----------------------------------------------------------------------------
# Conatus (controleur lent) : cue -> cle. Valeur = reponse (identite).
# ----------------------------------------------------------------------------
def init_conatus(d_k=D_K, d_in=D_IN, rng=rng):
    return rng.standard_normal((d_k, d_in)) / np.sqrt(d_in)


def encode_keys(Wk, cues):
    """cues (N,d_in) -> cles normalisees (N,d_k)."""
    return l2norm(cues @ Wk.T)


# ----------------------------------------------------------------------------
# Memoire a poids rapides F  (les 'lire' / 'inscrire')
# ----------------------------------------------------------------------------
def delta_write(S, K, V, beta=BETA):
    """Inscrire : ecriture sequentielle par delta rule (correction d'erreur)."""
    S = S.copy()
    for i in range(K.shape[0]):
        pred = K[i] @ S                      # prediction courante (d_v,)
        S = S + beta * np.outer(K[i], V[i] - pred)
    return S


def read(S, Q):
    """Lire : lecture associative o = q . F."""
    return Q @ S


def recall_metrics(O, V_pool):
    """Top-1 (cosinus, plus proche) ET signal moyen (cosinus a la vraie reponse).
    Le signal est continu : il s'effondre proprement quand la trace est effacee,
    la ou le top-1 sur du bruit reste un artefact d'argmax."""
    On, Vn = l2norm(O), l2norm(V_pool)
    sims = On @ Vn.T
    acc = float((sims.argmax(axis=1) == np.arange(O.shape[0])).mean())
    signal = float(np.mean(np.sum(On * Vn, axis=1)))
    return acc, signal


def build_engram(Wk, cues, responses, beta=BETA):
    """Construit l'engramme F d'une competence a partir d'un etat nul."""
    S0 = np.zeros((D_K, D_V))
    K = encode_keys(Wk, cues)
    return delta_write(S0, K, responses, beta), K


# ----------------------------------------------------------------------------
# Les operations inedites : transferer / oublier / gouverner
# ----------------------------------------------------------------------------
def low_rank(S, r):
    """Transferer : tronque l'engramme au rang r (delta compact)."""
    U, s, Vt = np.linalg.svd(S, full_matrices=False)
    return (U[:, :r] * s[:r]) @ Vt[:r]


def forget_projection(S, K_x):
    """Oublier : retire du F la composante lue par les cles de la competence X."""
    P = K_x.T @ np.linalg.pinv(K_x @ K_x.T) @ K_x   # projecteur sur span(cles X)
    return S - P @ S


def orthobasis(K):
    """Base orthonormale (d_k x r) de l'espace engendre par les cles (lignes de K)."""
    _, s, Vt = np.linalg.svd(K, full_matrices=False)
    r = int((s > 1e-8).sum())
    return Vt[:r].T


def govern_projection(dS, U):
    """Gouvernance : n'admet l'engramme que dans le sous-espace consenti U."""
    P = U @ U.T
    return P @ dS


# ----------------------------------------------------------------------------
# Entrainement du conatus : inscrire des engrammes non interferents
#   perte = || K K^T - I ||_F^2  (cles orthonormales -> faible diaphonie)
# ----------------------------------------------------------------------------
def ortho_loss_and_grad(Wk, cues):
    Z = cues @ Wk.T                          # (N, d_k)
    norm = np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9
    K = Z / norm                             # cles normalisees
    N = K.shape[0]
    G = K @ K.T - np.eye(N)
    loss = float((G * G).sum())
    dK = 4.0 * G @ K                         # dL/dK
    # backprop a travers la normalisation L2 (par ligne)
    dot = np.sum(dK * K, axis=1, keepdims=True)
    dZ = (dK - K * dot) / norm
    dWk = dZ.T @ cues                        # dL/dWk
    return loss, dWk


def train_conatus(Wk, steps=800, batch=16, lr=0.08, rng=rng):
    """Le conatus apprend a inscrire des engrammes non interferents, sur une
    plage de tailles N (capacite variable)."""
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
# Verification du gradient (garde-fou contre une derivation erronee)
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
# Outils statistiques et d'alignement
# ============================================================================
def agg(vals):
    a = np.array(vals, dtype=float)
    return float(a.mean()), float(a.std())


def pm(m, s, pct=True):
    return f"{m*100:5.1f} +/- {s*100:4.1f}" if pct else f"{m:5.2f} +/- {s:4.2f}"


def align_keyspace(Wk_src, Wk_dst, anchors):
    """Apprend W tel que (cles de dst) @ W approche (cles de src), a partir d'un
    jeu d'ancres partagees. Permet a un agent dst d'adresser l'engramme d'un
    agent src malgre un espace de cles different (Wk_src != Wk_dst)."""
    K_src = encode_keys(Wk_src, anchors)
    K_dst = encode_keys(Wk_dst, anchors)
    W, *_ = np.linalg.lstsq(K_dst, K_src, rcond=None)   # K_dst @ W ~= K_src
    return W


# ============================================================================
# Une experience = une graine. On agrege ensuite moyenne +/- ecart-type.
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
    """A et B ont des espaces de cles differents. On compare transfert naif vs
    transfert avec alignement appris sur des ancres partagees."""
    rt = np.random.default_rng(seed)
    Wk_A = init_conatus(rng=rt)
    Wk_B = init_conatus(rng=rt)
    cues_X, resp_X = sample_skill(rt)
    S_A, K_XA = build_engram(Wk_A, cues_X, resp_X)
    K_XB = encode_keys(Wk_B, cues_X)
    homo = recall_metrics(read(S_A, K_XA), resp_X)[0]            # meme Wk : borne sup
    naive = recall_metrics(read(S_A, K_XB), resp_X)[0]           # Wk different, sans alignement
    W = align_keyspace(Wk_A, Wk_B, rt.standard_normal((n_anchor, D_IN)))
    aligned = recall_metrics(read(S_A, K_XB @ W), resp_X)[0]     # avec alignement
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
    """Recouvrement controle des sous-espaces de cles entre X et Y, puis oubli de
    X : montre la degradation gracieuse de la retention de Y."""
    overlaps = [0.0, 0.25, 0.5, 0.75, 1.0]
    out = {ov: [] for ov in overlaps}
    for sd in seeds:
        rt = np.random.default_rng(sd)
        cues_X, resp_X = sample_skill(rt)
        S_A, K_X = build_engram(Wk, cues_X, resp_X)
        P = cues_X.T @ np.linalg.pinv(cues_X @ cues_X.T) @ cues_X   # proj sur span(cues X)
        for ov in overlaps:
            fresh = rt.standard_normal((N_PAIRS, D_IN))
            cues_Y = (1 - ov) * fresh + ov * (fresh @ P)           # Y plus ou moins dans X
            resp_Y = l2norm(rt.standard_normal((N_PAIRS, D_V)))
            S_Y, K_Y = build_engram(Wk, cues_Y, resp_Y)
            S_f = forget_projection(S_A + S_Y, K_X)
            out[ov].append(recall_metrics(read(S_f, K_Y), resp_Y)[0])
    return overlaps, out


def noise_sweep(Wk, seeds, r=8):
    """Corruption gaussienne de l'engramme transmis (canal bruite / empoisonnement
    leger) : sensibilite du rappel au bruit."""
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
# Pilote
# ============================================================================
def main():
    N_SEEDS = 20
    seeds = list(range(N_SEEDS))
    print("=" * 70)
    print(f"ENGRAMMIQUE DISTRIBUEE -- prototype  ({N_SEEDS} graines, moyenne +/- ecart-type)")
    print("=" * 70)

    rel = gradient_check()
    print(f"\n[verif gradient] erreur relative vs differences finies : {rel:.2e}"
          f"  ({'OK' if rel < 1e-5 else 'A REVOIR'})")

    # --- Conatus : capacite -------------------------------------------------
    Wk0 = init_conatus()
    Wk = train_conatus(Wk0.copy())
    print("\n--- Conatus : apprendre a inscrire (rappel vs charge) -----------")
    print(f"{'N assoc.':>9} | {'init':>16} | {'entraine':>16}")
    for n in [8, 16, 24, 32]:
        a0, _ = mean_recall(Wk0, n_pairs=n, n_skills=120, rng=np.random.default_rng(100 + n))
        a1, _ = mean_recall(Wk, n_pairs=n, n_skills=120, rng=np.random.default_rng(100 + n))
        print(f"{n:>9} | {a0*100:13.1f}%  | {a1*100:13.1f}%")

    # --- C1 : persistance (deterministe) ------------------------------------
    cx, rx = sample_skill(np.random.default_rng(0))
    S0, KX0 = build_engram(Wk, cx, rx)
    np.save("/tmp/F_state.npy", S0); S1 = np.load("/tmp/F_state.npy")
    print("\n--- C1  Persistance ---------------------------------------------")
    print(f"rappel avant/apres save-load : {recall_metrics(read(S0,KX0),rx)[0]*100:.0f}%"
          f" / {recall_metrics(read(S1,KX0),rx)[0]*100:.0f}%"
          f"   (ecart max sur F : {np.abs(S0-S1).max():.1e})")

    # --- Coeur multi-graines : C2, C3, C4 -----------------------------------
    trials = [run_core_trial(Wk, s) for s in seeds]
    st = {k: agg([t[k] for t in trials]) for k in trials[0]}

    print("\n--- C2  Transfert sans gradient (regime controle) ---------------")
    print("A possede X ; B possede deja Y ; B integre un delta de rang r.")
    print(f"{'rang r':>7} | {'cout (floats)':>13} | {'rappel X chez B':>18}")
    for r in (1, 2, 4, 8):
        print(f"{r:>7} | {r*(D_K+D_V):>13} | {pm(*st[f'tx{r}']):>18}")
    print(f"Y preserve (au rang 4) : {pm(*st['ty4'])} %")
    print(f"Reference replay brut : {N_PAIRS*(D_IN+D_V)} floats. "
          f"Gain seulement si la competence est compressible (ici rang <= 4).")

    print("\n--- C3  Oubli cible (substrat rapide) ---------------------------")
    print(f"signal de X apres projection : {pm(*st['forget_sigX'], pct=False)} "
          f"(s'effondre vers 0)")
    print(f"retention de Y (sous-espaces ~disjoints) : {pm(*st['forget_retY'])} %")

    print("\n--- C4  Gouvernance (consentement = sous-espace admis) ----------")
    print(f"engramme dans le sous-espace admis : {pm(*st['gov_admit'])} %")
    print(f"engramme hors du sous-espace admis : {pm(*st['gov_deny'])} %")

    # --- Hetero : agents a espaces de cles differents ------------------------
    hs = [run_hetero_trial(s) for s in seeds]
    homo = agg([h[0] for h in hs]); naive = agg([h[1] for h in hs])
    aligned = agg([h[2] for h in hs])
    print("\n--- Agents heterogenes  Wk_A != Wk_B ----------------------------")
    print(f"meme espace de cles (borne sup)   : {pm(*homo)} %")
    print(f"transfert naif (sans alignement)  : {pm(*naive)} %   <- le delta n'a pas de sens")
    print(f"avec alignement appris (128 ancres): {pm(*aligned)} %")

    # --- Sensibilites -------------------------------------------------------
    alphas, asweep = alpha_sweep(Wk, seeds)
    print("\n--- Sensibilite au facteur d'integration alpha (rang 4) ---------")
    for a in alphas:
        mx, sx = agg(asweep[a][0]); my, sy = agg(asweep[a][1])
        print(f"alpha={a:>4} : X chez B {pm(mx,sx)} %   |  Y preserve {pm(my,sy)} %")

    overlaps, osweep = overlap_sweep(Wk, seeds)
    print("\n--- Oubli vs recouvrement des sous-espaces (retention de Y) ------")
    for ov in overlaps:
        print(f"recouvrement={ov:>4} : retention Y {pm(*agg(osweep[ov]))} %")

    sigmas, nsweep = noise_sweep(Wk, seeds)
    print("\n--- Robustesse au bruit sur l'engramme transmis (rang 8) --------")
    for sg in sigmas:
        print(f"sigma={sg:>4} : rappel X {pm(*agg(nsweep[sg]))} %")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
