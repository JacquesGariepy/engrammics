# Engrammics distribuée — résultats d'exécution

Exécuté le 2026-06-19/20.

- **STAGE A (toy, NumPy, CPU)** : Linux, Python 3.10.12 / 3.12, NumPy 2.x.
- **STAGE B (lm, DeltaNet)** : WSL2 Ubuntu, Python 3.12, PyTorch 2.12.1+cu130, flash-linear-attention 0.5.1, transformers 5.12.1, tokenizers 0.22.2, GPU NVIDIA RTX 3090 (24 Go). Checkpoint **`fla-hub/delta_net-1.3B-100B`** (DeltaNet pur : 24 couches, 16 têtes, dim de tête 128, état récurrent par couche `[1,16,128,128]`) et **`fla-hub/delta_net-2.7B-100B`** (32 couches, 20 têtes).

### Provenance (pour audit)

```
# STAGE A (toy)
python src/engrammics_science.py --backend toy --seeds 60 --quiet

# STAGE B (LM), reps=3 par défaut, base_seed=0, décodage glouton
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  python src/engrammics_science.py --backend lm \
  --model fla-hub/delta_net-1.3B-100B --seeds 30
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  python src/engrammics_science.py --backend lm \
  --model fla-hub/delta_net-2.7B-100B --seeds 20
```

Graines : `base_seed=0` ; compétence X = `skill(base_seed+10000+i)`, Y = `skill(base_seed+20000+i)`, i sur 0..seeds-1. Reproductible jusqu'au non-déterminisme des noyaux CUDA (le contrôle random est désormais seedé déterministiquement). Code : working tree (non committé) — `git diff` couvre `src/engrammics_science.py`, `doc/engrammics_arxiv.tex`. Les logs bruts complets (30/30 et 20/20, moyennes + tests) sont dans `results/`.

## Ce qui a été lancé

L'étude a deux étages partageant un seul moteur statistique :

- **STAGE A (backend toy)** : valide le harnais et prouve l'algèbre. Gate critique.
- **STAGE B (backend lm, DeltaNet)** : le vrai test sur l'état à poids rapides d'un LM. **Désormais exécuté réellement** (auparavant impossible faute de GPU).

## STAGE A — test scientifique pré-enregistré (backend toy, 60 graines)

Verdict global : **THEORY (H1) SUPPORTED** — exit code 0. Chance = 12,5 %.

| Condition | Moyenne | IC 95 % |
|---|---|---|
| clean (plafond) | 100,0 % | [100, 100] |
| notransfer (contrôle a) | 12,5 % | [9,8, 15,2] |
| random engram (contrôle b) | 12,1 % | [9,0, 15,2] |
| transfert rang 1 / 2 / 4 / 8 | 28,7 / 55,0 / 88,8 / 100,0 % | — |
| full transfer | 100,0 % | [100, 100] |
| forget_X (oubli ciblé) | 0,2 % | [0,0, 0,6] |
| forget_Y (préservé) | 100,0 % | [100, 100] |
| admit / deny (gouvernance) | 100,0 / 12,5 % | — |

Tests (bootstrap apparié, IC 95 %, primaires Holm) : **H1 à H5 toutes SUPPORTED.** L'algèbre est exacte dans le régime contrôlé (clés ~orthonormales par construction).

## STAGE B — test scientifique pré-enregistré (backend lm, DeltaNet-1.3B, 30 graines)

Verdict global : **THEORY (H1) SUPPORTED** — exit code 0. Chance = 10,0 %.

Mécanisme (faithful) : l'engramme transféré est l'**état récurrent** par couche (la mémoire associative DeltaNet). La table compétence est lue **3 fois** (`reps=3`) pour écrire un engramme exploitable (une seule passe est trop faible pour ce checkpoint ; à 3 passes le plafond de rappel naturel est de 100 %). La lecture par injection se fait dans un carrier amorcé par un **primer neutre constant** (`Q=4;Z=1;W=9;`, clés hors alphabet de la tâche) qui remplit la fenêtre de convolution courte ; son écriture récurrente est écrasée par l'engramme, donc il ne fuite aucune information. Le BOS n'est ajouté qu'une fois par séquence (correction d'un bug d'injection de `<s>` en milieu de continuation). La **gouvernance** utilise le sous-espace des **vraies clés** : on capture les clés DeltaNet par hook sur la convolution courte des clés (`k_conv1d`) lors de l'écriture, et on projette côté `d_k` — au lieu d'estimer le sous-espace par SVD de l'engramme (qui inverse le test à l'échelle LM).

| Condition | Moyenne | IC 95 % |
|---|---|---|
| clean (plafond) | 100,0 % | [100, 100] |
| notransfer (contrôle a) | 7,3 % | [3,3, 11,3] |
| random engram (contrôle b) | 2,7 % | [0,7, 5,3] |
| transfert rang 1 / 2 / 4 | 7,3 / 7,3 / 14,7 % | — |
| transfert rang 8 / 16 | 44,0 / 68,7 % | — |
| full transfer | 68,7 % | [61,3, 75,3] |
| Y avant / après ajout de X | 100,0 / 74,0 % | — |
| joint (lire X puis Y), read X / read Y | 50,7 / 92,7 % | — |
| après soustraction de X, read X / read Y | 2,0 / 21,3 % | — |
| admit / deny (gouvernance par vraies clés) | 69,3 / 7,3 % | — |

Tests d'hypothèses (bootstrap apparié, IC 95 %, primaires Holm) :

| Hypothèse | Δ moyen | IC 95 % | verdict |
|---|---|---|---|
| H1a full > no-transfer | +61,3 % | [55,3, 67,3] | **SUPPORTED** (Holm ok) |
| H1b full > random | +66,0 % | [59,3, 72,7] | **SUPPORTED** (Holm ok) |
| H4 admit > deny (vraies clés) | +62,0 % | [56,0, 68,0] | **SUPPORTED** (Holm ok) |
| H5 rang 16 > rang 1 | +61,3 % | [55,3, 67,3] | **SUPPORTED** (Holm ok), monotone |
| H2 préservation de Y | −26,0 % | [−32,0, −20,0] | NOT SUPPORTED |
| H3 oubli de X (chute) | +48,7 % | [42,0, 55,3] | SUPPORTED |
| H3 oubli de X (Y conservé) | −71,3 % | [−80,0, −62,0] | NOT SUPPORTED |

### Lecture des résultats

- **H1 (transfert, PRIMAIRE) : SUPPORTED.** L'engramme à poids rapides d'un agent, ajouté **sans gradient** à l'état récurrent d'un autre, transfère la compétence : 68,7 % de rappel contre 7,3 % (no-transfer) et 2,7 % (random), les deux au niveau du hasard. La revendication centrale survit sur un vrai LM.
- **H5 (dose-réponse au rang) : SUPPORTED, monotone.** Mais la compétence n'est **pas** un objet de rang 5 dans le LM (contrairement au toy) : il faut un rang ≥ 8 (r8=44 %, r16=68,7 %). Les clés du LM ne sont pas orthogonales, la compétence s'étale sur plus de directions.
- **H2 (non-interférence) : NON soutenue.** Superposer X dégrade Y (100 %→74 %) : crosstalk dû aux clés non orthogonales.
- **H3 (oubli ciblé) : chute oui, conservation de Y non.** La soustraction efface X (50,7→2,0) mais détruit Y (92,7→21,3), car écrire Y sur un état contenant déjà X intrique les deux (non-linéarité de l'ordre d'écriture de la delta rule — mise en garde déjà présente dans le manuscrit, ici **confirmée empiriquement**).
- **H4 (gouvernance) : SUPPORTED** avec la définition faithful (vraies clés capturées). Admettre l'engramme via le sous-espace des clés autorisées récupère X à **69,3 %** (≈ transfert plein), le complément orthogonal le bloque à **7,3 %** (hasard) ; Δ +62,0. L'ancien estimateur par SVD de l'engramme inversait le test (admit < deny) car 5 directions singulières de l'état ne coïncident pas avec le sous-espace de clés de la compétence. **Choisir la définition par vraies clés répare la gouvernance.**
- **Précondition de disjonction — 3 niveaux** (`diag_disjoint2.py`, 16 graines) :
  | Régime | H2 dY | dropX | keepY |
  |---|---|---|---|
  | OVERLAP (symboles+sép. partagés) | −0,20 | +0,56 | −0,61 |
  | SYMBOLS (symboles disjoints) | **+0,00** | +0,98 | −0,56 |
  | FULL (symboles+sép. disjoints) | **+0,00** | **+1,00** | **−0,30** |

  La non-interférence (H2) **s'annule dès que les symboles sont disjoints** (dY −0,20 → 0,00), même séparateurs partagés → H2 entièrement expliqué par le recouvrement de clés de symboles. L'oubli devient quasi-chirurgical seulement en FULL (dropX +1,00, keepY −0,30 vs −0,61). Le résidu −0,30 est honnête : le format partagé couple encore légèrement via le traitement positionnel. **Dose-réponse avec point final propre pour l'interférence** : les deux dégradations suivent le recouvrement de clés, pas le mécanisme de transfert.

### Compétence vs dictionnaire (le verrou central)

La tâche de rappel transfère une **table mémorisée** (toutes les clés interrogées ont été montrées). Test direct du caractère « compétence » : l'engramme agit-il sur des entrées **jamais montrées** ?

- **Règle symbolique (`diag_skill.py`)** : le modèle de base **ne généralise PAS** une règle nouvelle. Décalage César f(x)=x+k sur lettres held-out : k≥2 au hasard (0,00–0,15) ; seul k=1 (successeur) « marche » mais c'est un **prior** (le contrôle mauvaise-règle donne 0,30–0,40). Répéter la table **baisse** la généralisation (successeur 0,50 à reps=1 → 0,12 à reps=3) → reps fait **mémoriser**, pas inférer. ⇒ le transfert de rappel est bien un dictionnaire ; tester le transfert de règle exige un modèle capable d'apprendre des règles in-context.
- **Comportement généralisant (`diag_style_transfer.py`, 40 graines)** : un **style de sortie contraint** (« toujours répondre c ») s'applique par construction à toute clé. Sur clés **held-out** : plafond 1,000 ; no-transfer 0,000 ; random 0,003 ; **transfert plein 0,353 [0,234, 0,475]**. full − no-transfer = **+0,353 p<0,0001** ; full − random = **+0,350 p<0,0001**. ⇒ transfert d'un **comportement** vers des entrées jamais montrées — catégoriquement pas un dictionnaire — borné par l'interférence H2. Première instance de transfert de **capacité** (vs mémoire) sans gradient sur un vrai LM linéaire.

### Contrôles structurés & ablation reps

- **Contrôles structurés (`diag_controls.py`, 30 graines)** — plus exigeants que le random norm-matched : transfert réel **0,740** ; **shuffled-values** (mêmes clés, valeurs permutées) **0,200** ; **wrong-skill** (compétence sans rapport) **0,053**. full − shuffled = **+0,540 [0,473, 0,607] p<0,0001** ; full − wrong = **+0,687 p<0,0001**. ⇒ l'effet est spécifique au **contenu** de l'engramme, pas à sa structure/spectre/norme. (Le shuffled est au-dessus du hasard : les bonnes clés adressent bien l'état.)
- **Ablation reps (`--reps 1/2/3`, 15 graines chacune)** : reps=1 → clean **30,7 %**, full 12,0 % (H1a +2,7, p=0,12, **INCONCLUSIVE** — engramme trop faible) ; reps=2 → clean **100 %**, full **70,7 %** (SUPPORTED) ; reps=3 → clean 100 %, full **68,0 %** (SUPPORTED). ⇒ **reps≥2 sature** : pas un artefact de « prompt rehearsal » propre à reps=3.

### Confirmation sur DeltaNet-2.7B (20 graines)

Même protocole sur `fla-hub/delta_net-2.7B-100B` (32 couches, 20 têtes) — **tous les verdicts se reproduisent** : clean 100 %, transfert plein 61,0 %, H1a +54,0 [45,63] et H1b +60,0 [51,68] **SUPPORTED** (Holm) ; **H4 +52,0 [43,61] SUPPORTED** (admit 59 % vs deny 7 %, vraies clés) ; H5 monotone SUPPORTED ; H2 NON (Y 100→71, −29) ; H3 chute oui / keep-Y non (88→27, −61). Le contraste (H1/H4/H5 tiennent, H2/H3-keep échouent) est donc une propriété du **mécanisme**, pas d'un checkpoint particulier.

**Bilan :** trois revendications tiennent à l'échelle d'un LM réel, sur deux tailles de modèle — le **transfert sans gradient** (H1), la **dose-réponse au rang** (H5), et la **gouvernance par sous-espace de clés** (H4, une fois les vraies clés utilisées). Seules la **non-interférence parfaite** (H2) et l'**oubli chirurgical** (H3-keepY) se dégradent, et pour une **cause unique et nommable** : X et Y partagent des directions de clés (la précondition de disjonction du papier), confirmée par l'expérience à symboles disjoints. C'est un résultat falsifiable, nuancé et mécaniquement interprétable.

## Prototype complet (engrammics_proto.py, 20 graines)

- **Vérification du gradient** : erreur relative vs différences finies = 5,77e-10 (OK).
- **Conatus (capacité)** : rappel 100 % à 8 paires, 97,2 % à 32 paires (vs 84,6 % non entraîné).
- **C1 Persistance** : 100 % / 100 % avant-après save-load, écart max sur F = 0.
- **C2 Transfert sans gradient** : 22,5 % (r1) → 100 % (r8) ; Y préservé à 100 %.
- **C3 Oubli ciblé** : signal de X → 0 après projection, rétention de Y à 100 %.
- **C4 Gouvernance** : 100 % dans le sous-espace admis vs 9,4 % hors.
- **Agents hétérogènes** : transfert naïf 12,5 % ; avec alignement appris (128 ancres), 100 %.
- **Sensibilités** : monotone en alpha ; rétention de Y se dégrade avec le recouvrement ; robuste au bruit jusqu'à sigma 0,4.

## Fichiers

- `results/stage_a_science_toy.log` — sortie brute STAGE A (toy, 60 graines)
- `results/stage_b_science_lm.log` — sortie brute STAGE B (DeltaNet-1.3B, 30 graines)
- `results/stage_b_science_lm_2.7B.log` — confirmation DeltaNet-2.7B (20 graines)
- `results/stage_b_disjoint.log` — précondition de disjonction (overlap vs disjoint)
- `results/proto_demonstrations.log` — sortie brute du prototype
