# Engrammics distribuée : résultats d'exécution

Exécuté le 2026-06-19/20.

- **STAGE A (toy, NumPy, CPU)** : Linux, Python 3.10.12 / 3.12, NumPy 2.x.
- **STAGE B (lm, DeltaNet)** : WSL2 Ubuntu, Python 3.12, PyTorch 2.12.1+cu130, flash-linear-attention 0.5.1, transformers 5.12.1, tokenizers 0.22.2, GPU NVIDIA RTX 3090 (24 Go). Checkpoint **`fla-hub/delta_net-1.3B-100B`** (DeltaNet pur : 24 couches, 16 têtes, dim de tête 128, état récurrent par couche `[1,16,128,128]`) et **`fla-hub/delta_net-2.7B-100B`** (32 couches, 20 têtes).

### Provenance (pour audit)

```
# STAGE A (toy)
python src/engrammics_science.py --backend toy --seeds 60 --quiet

# STAGE B (LM), reps=3 par défaut, base_seed=0, décodage glouton
#   --dump écrit un CSV par-seed de toutes les conditions (audit)
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  python src/engrammics_science.py --backend lm \
  --model fla-hub/delta_net-1.3B-100B --seeds 30 \
  --dump results/perseed_lm_1.3B.csv
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  python src/engrammics_science.py --backend lm \
  --model fla-hub/delta_net-2.7B-100B --seeds 20 \
  --dump results/perseed_lm_2.7B.csv
```

**Environnement** : WSL2 Ubuntu, Python 3.12, torch 2.12.1+cu130, flash-linear-attention 0.5.1, transformers 5.12.1, tokenizers 0.22.2, GPU RTX 3090. Checkpoints `fla-hub/delta_net-{1.3B,2.7B}-100B` (DeltaNet pur).

Graines : `base_seed=0` ; compétence X = `skill(base_seed+10000+i)`, Y = `skill(base_seed+20000+i)`, i sur 0..seeds-1. Reproductible jusqu'au non-déterminisme des noyaux CUDA (le contrôle random est seedé déterministiquement). Les logs bruts complets (30/30 et 20/20, moyennes + tests) et les **CSV par-seed** (`results/perseed_lm_*.csv`) sont fournis : chaque moyenne/IC du papier se recalcule depuis les valeurs par-seed. Commit de base : voir `git log` du dépôt.

## Ce qui a été lancé

L'étude a deux étages partageant un seul moteur statistique :

- **STAGE A (backend toy)** : valide le harnais et prouve l'algèbre. Gate critique.
- **STAGE B (backend lm, DeltaNet)** : le vrai test sur l'état à poids rapides d'un LM. **Désormais exécuté réellement** (auparavant impossible faute de GPU).

## STAGE A : test scientifique pré-enregistré (backend toy, 60 graines)

Verdict global : **THEORY (H1) SUPPORTED**, exit code 0. Chance = 12,5 %.

| Condition | Moyenne | IC 95 % |
|---|---|---|
| clean (plafond) | 100,0 % | [100, 100] |
| notransfer (contrôle a) | 12,5 % | [9,8, 15,2] |
| random engram (contrôle b) | 12,1 % | [9,0, 15,2] |
| transfert rang 1 / 2 / 4 / 8 | 28,7 / 55,0 / 88,8 / 100,0 % | n/a |
| full transfer | 100,0 % | [100, 100] |
| forget_X (oubli ciblé) | 0,2 % | [0,0, 0,6] |
| forget_Y (préservé) | 100,0 % | [100, 100] |
| admit / deny (gouvernance) | 100,0 / 12,5 % | n/a |

Tests (bootstrap apparié, IC 95 %, primaires Holm) : **H1 à H5 toutes SUPPORTED.** L'algèbre est exacte dans le régime contrôlé (clés ~orthonormales par construction).

## STAGE B : test scientifique pré-enregistré (backend lm, DeltaNet-1.3B, 30 graines)

Verdict global : **THEORY (H1) SUPPORTED**, exit code 0. Chance = 10,0 %.

Mécanisme (faithful) : l'engramme transféré est l'**état récurrent** par couche (la mémoire associative DeltaNet). La table compétence est lue **3 fois** (`reps=3`) pour écrire un engramme exploitable (une seule passe est trop faible pour ce checkpoint ; à 3 passes le plafond de rappel naturel est de 100 %). La lecture par injection se fait dans un carrier amorcé par un **primer neutre constant** (`Q=4;Z=1;W=9;`, clés hors alphabet de la tâche) qui remplit la fenêtre de convolution courte ; son écriture récurrente est écrasée par l'engramme, donc il ne fuite aucune information. Le BOS n'est ajouté qu'une fois par séquence (correction d'un bug d'injection de `<s>` en milieu de continuation). La **gouvernance** utilise le sous-espace des **vraies clés** : on capture les clés DeltaNet par hook sur la convolution courte des clés (`k_conv1d`) lors de l'écriture, et on projette côté `d_k`, au lieu d'estimer le sous-espace par SVD de l'engramme (qui inverse le test à l'échelle LM).

| Condition | Moyenne | IC 95 % |
|---|---|---|
| clean (plafond) | 100,0 % | [100, 100] |
| notransfer (contrôle a) | 7,3 % | [3,3, 11,3] |
| random engram (contrôle b) | 2,7 % | [0,7, 5,3] |
| transfert rang 1 / 2 / 4 | 7,3 / 7,3 / 14,7 % | n/a |
| transfert rang 8 / 16 | 44,0 / 68,7 % | n/a |
| full transfer | 68,7 % | [61,3, 75,3] |
| Y avant / après ajout de X | 100,0 / 74,0 % | n/a |
| joint (lire X puis Y), read X / read Y | 50,7 / 92,7 % | n/a |
| après soustraction de X, read X / read Y | 2,0 / 21,3 % | n/a |
| admit / deny (gouvernance par vraies clés) | 69,3 / 7,3 % | n/a |

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
- **H3 (oubli ciblé) : chute oui, conservation de Y non.** La soustraction efface X (50,7→2,0) mais détruit Y (92,7→21,3), car écrire Y sur un état contenant déjà X intrique les deux (non-linéarité de l'ordre d'écriture de la delta rule, mise en garde déjà présente dans le manuscrit, ici **confirmée empiriquement**).
- **H4 (gouvernance) : SUPPORTED** avec la définition faithful (vraies clés capturées). Admettre l'engramme via le sous-espace des clés autorisées récupère X à **69,3 %** (≈ transfert plein), le complément orthogonal le bloque à **7,3 %** (hasard) ; Δ +62,0. L'ancien estimateur par SVD de l'engramme inversait le test (admit < deny) car 5 directions singulières de l'état ne coïncident pas avec le sous-espace de clés de la compétence. **Choisir la définition par vraies clés répare la gouvernance.**
- **Précondition de disjonction : 3 niveaux** (`diag_disjoint2.py`, 16 graines) :
  | Régime | H2 dY | dropX | keepY |
  |---|---|---|---|
  | OVERLAP (symboles+sép. partagés) | −0,20 | +0,56 | −0,61 |
  | SYMBOLS (symboles disjoints) | **+0,00** | +0,98 | −0,56 |
  | FULL (symboles+sép. disjoints) | **+0,00** | **+1,00** | **−0,30** |

  La non-interférence (H2) **s'annule dès que les symboles sont disjoints** (dY −0,20 → 0,00), même séparateurs partagés → H2 entièrement expliqué par le recouvrement de clés de symboles. L'oubli devient quasi-chirurgical seulement en FULL (dropX +1,00, keepY −0,30 vs −0,61). Le résidu −0,30 est honnête : le format partagé couple encore légèrement via le traitement positionnel. **Dose-réponse avec point final propre pour l'interférence** : les deux dégradations suivent le recouvrement de clés, pas le mécanisme de transfert.

- **Réparation de H2 à l'injection : SANS retraining (`diag_ortho_inject.py`, 30 graines)** : au lieu d'ajouter `eX` brut, on l'ajoute projeté orthogonalement aux clés de l'hôte : `eX' = (I − P_Y) eX`. Résultat : **dommage à l'hôte = 0,00** (Y reste à 1,000 vs naïf 0,673, dommage −0,327) ; coût : fidélité du transfert X passe de 0,747 à 0,413 (on garde la part de X disjointe de Y). ⇒ la non-interférence n'est **pas perdue mais achetable** : compromis réglable préservation/fidélité, **avec la même machinerie de gouvernance** (le projecteur de consentement sert aussi d'allocateur anti-collision).

- **Réparation de H3 à l'oubli : Y-safe, SANS retraining (`diag_h3_fix.py`, 24 graines, régime difficile alphabet partagé)** : oublier X en ne retirant que ses directions de clés **orthogonales à Y** (`P_{X\Y}`). joint X=0,550 Y=0,867 ; soustraction naïve dropX +0,533 mais **keepY −0,625** (Y détruit) ; projection complète keepY −0,658 ; **projection X-only dropX +0,217, keepY +0,067** (Y préservé : 0,867→0,933). ⇒ dual exact de H2 : oubli **Y-safe** au prix d'un retrait partiel de X (part partagée subsiste). Les deux propriétés idéalisées (H2, H3) deviennent des **compromis réglables avec le même projecteur de clés** : deux limites deviennent une méthode.

- **Cross-checkpoint : RÉUSSI (`diag_xcheckpoint.py`, 25 graines + 24 ancres)** : transfert entre **deux checkpoints RWKV-7-1.5B vraiment différents** : `g1` (raisonnement) → `world` (base multilingue), mêmes dims, entraînements distincts, états récurrents compatibles mais bases de clés différentes. Les deux font la tâche (plafond 1,000).

  | Condition | Valeur |
  |---|---|
  | B (world) lit son propre engramme | 1,000 |
  | **naïf g1→world (sans alignement)** | **0,040** (≈ hasard) |
  | **aligné g1→world (W appris)** | **0,968** [0,936, 0,992] |
  | aligné − naïf | **+0,928 p<10⁻⁴** |

  ⇒ le transfert naïf entre checkpoints différents **échoue** (l'engramme est adressé dans la base de clés du donneur), mais l'**alignement linéaire appris** (`F_B ≈ W F_A`, par couche/tête, depuis 24 ancres partagées) le **récupère intégralement** (0,968 ≈ plafond de B). C'est le résultat hétérogène du jouet (Table 4) **reproduit à l'échelle LM, entre deux entraînements distincts**. Le scénario qui motive le papier est donc démontré. (Suppose même architecture/dims + ancres partageables ; cross-**architecture** reste ouvert.)
  - Tentative préalable `delta_net-1.3B-100B → delta_net-1.3B-8K-100B` abandonnée : le 8K ne fait pas la tâche (rappel propre 0,000), paire invalide.

### Compétence vs dictionnaire : le verrou central, FRANCHI

La tâche de rappel transfère une **table mémorisée**. Test direct du caractère « compétence » : l'engramme agit-il sur des entrées **jamais montrées** (held-out) ?

- **Règle symbolique arbitraire (`diag_skill.py`, 1.3B + 2.7B)** : les modèles **ne généralisent PAS** une règle nouvelle. César f(x)=x+k sur held-out : k≥2 au hasard aux deux tailles ; seul k=1 (successeur) « marche » mais c'est un **prior** (contrôle mauvaise-règle 0,30–0,40). Répéter la table **baisse** la généralisation → mémorisation, pas induction. ⇒ limite du **modèle**, pas du mécanisme : une règle arbitraire ne peut être transférée car le modèle ne sait pas l'apprendre in-context.

- **Règle de concept connue : transfert RÉUSSI (`diag_rule_transfer.py`)** : classification **voyelle/consonne** (assignation aléatoire voyelle→a, consonne→b). Les deux modèles l'appliquent à des lettres held-out in-context (0,73–0,79 vs wrong-label 0,21–0,27, `results/stage_b_concept_probe.log`). On écrit l'engramme de la règle, on l'injecte dans un receveur, on score sur lettres **jamais montrées** :

  | Condition (held-out) | 1.3B (35 graines) | 2.7B (52 graines) |
  |---|---|---|
  | engramme seul (plafond règle) | 0,746 | 0,752 |
  | no-transfer | 0,000 | 0,000 |
  | random | 0,025 | 0,007 |
  | **wrong-label** (étiquettes inversées) | 0,007 | 0,062 |
  | **transfert superposé** | **0,132** | **0,166** |
  | Δ vs no-transfer | +0,132 **p<10⁻⁴** | +0,166 **p<10⁻⁴** |
  | Δ vs random | +0,107 **p<10⁻⁴** | +0,159 **p<10⁻⁴** |
  | Δ vs wrong-label | +0,125 **p<10⁻⁴** | +0,103 **p<10⁻⁴** |

  (Seed-bump 2.7B 26→52 graines : la cellule wrong-label passe de p=0,003 à **p<10⁻⁴**.)

  ⇒ Injecté dans un receveur neutre, l'engramme **porte la règle** (classe le held-out à ~0,73–0,75 vs ~0 pour tous les contrôles). Superposé à une compétence existante, il transfère encore la règle au-dessus des **trois contrôles**, dont le **wrong-label** (même structure, assignation opposée), sur **deux tailles de modèle**. Borné sous le plafond par l'interférence H2, mais statistiquement net et mesuré sur held-out : **ce n'est pas un dictionnaire.** Première démonstration qu'une règle de classification généralisante circule via un engramme à poids rapides, sans gradient, vers une autre instance.

- **Comportement généralisant (`diag_style_transfer.py`, 40 graines)** : un **style de sortie contraint** (« toujours répondre c ») corrobore : held-out : plafond 1,000 ; no-transfer 0,000 ; random 0,003 ; **transfert 0,353** ; Δ +0,353/+0,350 **p<10⁻⁴**.

- **Réplication sur 2ᵉ architecture, RWKV-7 1.5B (`diag_rule_transfer.py`, 52 graines)** : le transfert voyelle/consonne marche sur RWKV-7 (état récurrent lu **sans modif**) : plafond 0,798 ; FULL **0,147** ; **Δ vs no-transfer +0,147 p<10⁻⁴**, **Δ vs wrong-label +0,113 p=0,0004** (contrôle de spécificité décisif). En revanche **Δ vs random = +0,067, p=0,05, IC [−0,01, 0,15]** (limite) : le random norm-matched monte à 0,08 sur cette tâche 2-classes, donc la marge sur le bruit additif est mince sur RWKV. ⇒ la règle transfère au sens qui compte (bat l'état propre du receveur ET un engramme à étiquettes inversées de même structure), mais on ne surclaime pas la séparation au random. Le seed-bump 31→52 a **affaibli** cette cellule (p=0,009 → 0,05) : honnêteté de l'analyse de puissance.

- **Induction de règle arbitraire : RWKV-7 2.9B (`diag_skill.py`)** : contrairement à DeltaNet (k≥2 au hasard), RWKV-7 2.9B **généralise** un César k=2 sur held-out : **held 0,33–0,44** vs wrong-rule 0,07–0,08 (k=3 ~0,17). Premier modèle linéaire de notre suite à induire une règle *nouvelle* en contexte.

- **TRANSFERT de règle ARBITRAIRE : RWKV-7 2.9B (`diag_caesar_transfer.py`, 50 graines)** : l'engramme d'un César k=2 *fraîchement appris*, injecté dans un **receveur neutre**, applique la transformation à des lettres **jamais montrées** :

  | Condition (César k=2, held-out) | Valeur |
  |---|---|
  | engramme seul (plafond règle) | **0,440** [0,385, 0,492] |
  | random engram seul | 0,005 |
  | wrong-shift engram seul | 0,080 |
  | clean − random | **+0,435 p<10⁻⁴** |
  | clean − wrong-shift | **+0,360 p<10⁻⁴** |

  ⇒ **transfert sans gradient d'une règle ARBITRAIRE généralisante** vers un receveur, sur un modèle capable de l'apprendre en contexte. Le contrôle **wrong-shift** (même format/structure, décalage différent) est décisif : à 0,08, il ne reproduit pas la cible → c'est bien la **règle** qui voyage, pas un prior ni la forme de l'engramme. Borné par le plafond in-context (~0,44). Caveat fla RWKV-7 non bit-exact.

- **FAMILLE de règles induites : RWKV-7 2.9B (`diag_rule_family.py`, 50 graines)** : pour réfuter le « n=1 », on teste 3 règles de plus (engramme seul → receveur neutre, held-out), chacune vs random ET vs **wrong-rule** (même structure, transformation différente = contrôle de spécificité décisif).

  | Règle | engramme seul | Δ vs random | Δ vs wrong-rule | spécifique ? |
  |---|---|---|---|---|
  | Caesar k=3 | 0,205 | +0,205 (p<10⁻⁴) | +0,145 (p<10⁻⁴) | ✅ |
  | Caesar k=5 (plus dur) | 0,075 | +0,070 (p<10⁻⁴) | +0,052 (p<10⁻⁴) | ✅ |
  | **Atbash** (réflexion x→25−x, **non-translation**) | 0,172 | +0,168 (p<10⁻⁴) | +0,122 (p<10⁻⁴) | ✅ |
  | digit:3 (autre domaine, x→(x+3) mod 10) | 0,107 | +0,107 (p<10⁻⁴) | **−0,030 (p=0,95)** | ❌ |

  ⇒ **4 règles induites distinctes** (k=2 ci-dessus + k=3, k=5, Atbash) de **2 types structurels** (translations + une **réflexion**) transfèrent spécifiquement → le claim « règle, pas dictionnaire » ne repose plus sur n=1. **Limite honnête** : sur le domaine **chiffres**, l'engramme bat le random mais **pas** le wrong-rule (le digit:7 fait aussi bien) → pas de transfert spécifique là ; le modèle n'induit pas proprement la règle modulaire sur 10 symboles.

### Tâche moins-jouet (multi-token) & baseline shuffled-key

- **`diag_multitoken.py`, 40 graines** : letter → **valeur 2 chiffres**, correcte seulement si les **deux tokens** matchent (hasard ~1/90). plafond 0,985 ; no-transfer 0,010 ; random 0,000 ; **shuffled-key** (clés+valeurs correctes, appariement permuté) **0,100** ; **FULL TRANSFER 0,635**. Δ vs no-transfer +0,625, vs random +0,635, **vs shuffled-key +0,535** (tous p<10⁻⁴). ⇒ (1) l'engramme porte de vraies **associations multi-token**, pas du single-token ; (2) le transfert dépend de l'**appariement clé↔valeur** spécifique (shuffled-key le casse), pas juste de la présence des bonnes clés/valeurs.

### Contrôles structurés & ablation reps

- **Contrôles structurés (`diag_controls.py`, 30 graines)** : plus exigeants que le random norm-matched : transfert réel **0,740** ; **shuffled-values** (mêmes clés, valeurs permutées) **0,200** ; **wrong-skill** (compétence sans rapport) **0,053**. full − shuffled = **+0,540 [0,473, 0,607] p<0,0001** ; full − wrong = **+0,687 p<0,0001**. ⇒ l'effet est spécifique au **contenu** de l'engramme, pas à sa structure/spectre/norme. (Le shuffled est au-dessus du hasard : les bonnes clés adressent bien l'état.)
- **Ablation reps (`--reps 1/2/3`, 15 graines chacune)** : reps=1 → clean **30,7 %**, full 12,0 % (H1a +2,7, p=0,12, **INCONCLUSIVE** : engramme trop faible) ; reps=2 → clean **100 %**, full **70,7 %** (SUPPORTED) ; reps=3 → clean 100 %, full **68,0 %** (SUPPORTED). ⇒ **reps≥2 sature** : pas un artefact de « prompt rehearsal » propre à reps=3.

### Confirmation sur DeltaNet-2.7B (20 graines)

Même protocole sur `fla-hub/delta_net-2.7B-100B` (32 couches, 20 têtes), **tous les verdicts se reproduisent** : clean 100 %, transfert plein 61,0 %, H1a +54,0 [45,63] et H1b +60,0 [51,68] **SUPPORTED** (Holm) ; **H4 +52,0 [43,61] SUPPORTED** (admit 59 % vs deny 7 %, vraies clés) ; H5 monotone SUPPORTED ; H2 NON (Y 100→71, −29) ; H3 chute oui / keep-Y non (88→27, −61). Le contraste (H1/H4/H5 tiennent, H2/H3-keep échouent) est donc une propriété du **mécanisme**, pas d'un checkpoint particulier.

**Bilan :** trois revendications tiennent à l'échelle d'un LM réel, sur deux tailles de modèle : le **transfert sans gradient** (H1), la **dose-réponse au rang** (H5), et la **gouvernance par sous-espace de clés** (H4, une fois les vraies clés utilisées). Seules la **non-interférence parfaite** (H2) et l'**oubli chirurgical** (H3-keepY) se dégradent, et pour une **cause unique et nommable** : X et Y partagent des directions de clés (la précondition de disjonction du papier), confirmée par l'expérience à symboles disjoints. C'est un résultat falsifiable, nuancé et mécaniquement interprétable.

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

Inventaire complet, mis en regard des éléments du papier (`doc/engrammics_arxiv.tex`). Chaque log est cité une fois dans `REPRODUCE.md` §4, qui donne la commande exacte qui l'a produit.

**Régime contrôlé (CPU, NumPy)**
- `results/stage_a_science_toy.log` : Tables `tab:means` + `tab:tests` (toy, 60 graines)
- `results/proto_demonstrations.log` : Tables `tab:capacity`, `tab:hetero`, sensibilités + vérif du gradient (20 graines)

**Langage-modèle : protocole principal (DeltaNet)**
- `results/stage_b_science_lm.log` + `results/perseed_lm_1.3B.csv` : Tables `tab:lm-means` + `tab:lm-tests` (1.3B, 30 graines, par-seed)
- `results/perseed_lm_2.7B.csv` : par-seed de la confirmation 2.7B
- `results/stage_b_science_lm_2.7B.log` : confirmation DeltaNet-2.7B (20 graines) : paragraphe « Robustness across model size »
- `results/stage_b_reps_ablation.log` : ablation reps=1/2/3 (15 graines chacune) : paragraphe « Specificity and the role of repetition »

**Dégradations et réparations (DeltaNet-1.3B)**
- `results/stage_b_disjoint2.log` : Table `tab:disjoint`, précondition de disjonction 3 niveaux (16 graines)
- `results/stage_b_disjoint.log` : précurseur 2 niveaux (16 graines) : développement-only, cité dans `REPRODUCE.md` §5
- `results/stage_b_controls.log` : contrôles structurés shuffled-values / wrong-skill (30 graines) : paragraphe « Specificity and the role of repetition »
- `results/stage_b_multitoken.log` : associations multi-token + contrôle shuffled-key (40 graines) : paragraphe « Multi-token associations »
- `results/stage_b_ortho_inject.log` : réparation H2 par injection orthogonale (30 graines) : paragraphe « Engineering non-interference at injection time »
- `results/stage_b_h3_fix.log` : réparation H3 par oubli X-only (24 graines) : paragraphe « Engineering Y-safe forgetting »

**Mémoire ou compétence ? (transfert de règle)**
- `results/stage_b_rule_probe.log` : sonde César, DeltaNet 1.3B/2.7B : paragraphe « Arbitrary symbolic rules »
- `results/stage_b_concept_probe.log` : sonde concept voyelle/consonne in-context (held 0,73–0,79 vs wrong-label 0,21–0,27), DeltaNet 1.3B/2.7B : paragraphe « A known-concept rule generalizes, and transfers »
- `results/stage_b_rule_probe_rwkv1.5B.log` : sondes César + concept, RWKV-7-1.5B
- `results/stage_b_rule_probe_rwkv2.9B.log` : sonde César, RWKV-7-2.9B : paragraphe « A non-trivial induced rule transfers »
- `results/stage_b_rule_transfer.log` : Table `tab:rule`, colonne 1.3B (voyelle/consonne, 35 graines)
- `results/stage_b_rule_transfer_2.7B.log` : Table `tab:rule`, colonne 2.7B (52 graines)
- `results/stage_b_rule_transfer_rwkv1.5B.log` : réplication RWKV-7-1.5B (52 graines) : paragraphe « Replication on a second architecture »
- `results/stage_b_style_transfer.log` : style de sortie constant (40 graines) : paragraphe « A constant output style also transfers »
- `results/stage_b_caesar_transfer_rwkv2.9B.log` : transfert César k=2 (50 graines) : paragraphe « A non-trivial induced rule transfers »
- `results/stage_b_xcheckpoint_rwkv.log` : transfert cross-checkpoint g1→world (25 graines, 24 ancres) : paragraphe « Transfer across distinct checkpoints »

**Scratch (non cités par le papier)**
- `results/.diag_disjoint.out`, `results/.diag_keys.out` : sorties de développement, voir `REPRODUCE.md` §5 (« stale scratch »)

**Scripts**
- `scripts/diag_*.py` : scripts de diagnostic (un par log ci-dessus) ; la correspondance papier↔exploratoire est dans `REPRODUCE.md` §5
- `src/engrammics_science.py` : le harnais (un moteur statistique, deux backends `toy`/`lm`)
- `src/engrammics_proto.py` : le prototype NumPy (capacité, hétérogénéité, sensibilités)
