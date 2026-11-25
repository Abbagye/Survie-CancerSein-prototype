"""Moteur de calcul du prototype (sans interface) : encodage, prédiction, intervalle d'incertitude et explication."""
import json
from pathlib import Path
import numpy as np

STADES = ["I-II", "III", "IV"]; GRADES = ["I-II", "III"]; SOUS_TYPES = ["Luminal", "Triple négatif", "HER2+"]

def charger(chemin="modele.json"):
    """Lit modele.json. Un nom simple est cherché à côté de ce fichier, quel que soit le dossier d'où l'on lance l'application."""
    p = Path(chemin)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent / p
    with open(p, "r", encoding="utf-8" ) as f:
        return json.load(f)

def encoder(critere, age, stade, grade, sous_type):
    """Transforme la saisie en vecteur de variables, dans l'ordre du modèle."""
    a, g = age / 10.0, float(grade == "III")
    tn, h2 = float(sous_type == "Triple négatif"), float(sous_type == "HER2+")
    if critere == "os":
        return np.array([a, float(stade == "IV"), g, tn, h2])
    return np.array([a, float(stade == "III"), float(stade == "IV"), g, tn, h2])

def predire(modele, critere, x):
    """Courbe de survie (avec IC 95 % bootstrap), groupe de risque et contributions des variables."""
    M = modele[critere]; beta = np.array(M["beta"]); H0 = np.array(M["H0"]); mois = np.array(M["grille_mois"])
    eta = float(x @ beta); S = np.exp(-H0 * np.exp(eta))
    Bb, Hb = np.array(M["boot_beta"]), np.array(M["boot_H0"])
    Sb = np.exp(-Hb * np.exp(Bb @ x)[:, None]); lo, hi = np.percentile(Sb, [2.5, 97.5], axis=0)
    q1, q2 = M["seuils"]; groupe = 0 if eta <= q1 else (1 if eta <= q2 else 2)
    phi = beta * (x - np.array(M["moyenne_variables"]))
    # les variables de stade (III, IV) sont codées par rapport au stade I-II : on cumule leur effet pour éviter des lectures trompeuses
    brut = list(zip(M["noms"], phi)); stade_phi = sum(p for n, p in brut if n.startswith("Stade"))
    groupees = [("Stade", stade_phi)] + [(n.replace(" (contre I-II)", ""), p) for n, p in brut if not n.startswith("Stade")]
    contributions = sorted(((n, float(np.exp(p))) for n, p in groupees), key=lambda c: abs(np.log(c[1])), reverse=True)
    def a(h): return dict(p=float(S[h]), bas=float(min(lo[h], S[h])), haut=float(max(hi[h], S[h])))
    return dict(mois=mois, S=S, bas=np.minimum(lo, S), haut=np.maximum(hi, S), a12=a(12), a24=a(24), a36=a(36), groupe=groupe,
                risque_relatif=float(np.exp(eta - M["eta_moyen"])), contributions=contributions)

def fr(v, d=2):
    return f"{v:.{d}f}".replace(".", ",")

def compte_rendu(modele, critere, age, stade, grade, sous_type, r):
    M = modele[critere]; lib = M["libelle"].lower()
    ag = [f"{n} (×{fr(m)})" for n, m in r["contributions"] if m > 1.05][:3]; pr = [f"{n} (×{fr(m)})" for n, m in r["contributions"] if m < 0.95][:3]
    t = (f"Patiente de {age} ans, stade {stade}, grade {grade}, sous-type {sous_type if sous_type == 'HER2+' else sous_type.lower()}.\n"
         f"Le modèle estime une probabilité de {lib} de {100*r['a12']['p']:.0f} % à 12 mois (IC 95 % {100*r['a12']['bas']:.0f}-{100*r['a12']['haut']:.0f} %), "
         f"{100*r['a24']['p']:.0f} % à 24 mois (IC 95 % {100*r['a24']['bas']:.0f}-{100*r['a24']['haut']:.0f} %) et {100*r['a36']['p']:.0f} % à 36 mois "
         f"(IC 95 % {100*r['a36']['bas']:.0f}-{100*r['a36']['haut']:.0f} %).\n"
         f"Son risque relatif est de ×{fr(r['risque_relatif'])} par rapport à la patiente moyenne de la cohorte ({M['groupes'][r['groupe']]['nom'].lower()}).\n")
    t += "Facteurs qui augmentent le risque : " + (", ".join(ag) if ag else "aucun facteur marqué") + ".\n"
    t += "Facteurs qui le réduisent : " + (", ".join(pr) if pr else "aucun facteur marqué") + ".\n"
    t += (f"Modèle de recherche construit sur {M['n']} patientes ({M['evenements']} événements) d'un seul centre, non validé cliniquement : "
          "ce compte-rendu ne doit pas guider une décision thérapeutique.")
    return t
