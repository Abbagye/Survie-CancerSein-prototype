import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from moteur import charger, encoder, predire, compte_rendu, fr, STADES, GRADES, SOUS_TYPES

st.set_page_config(page_title="Survie du cancer du sein : prototype d'aide à la décision", layout="wide")

@st.cache_data
def modele():
    return charger("modele.json")

M = modele()
COULEURS = ["#4F9D69", "#E0A030", "#C0504D"]

st.title("Survie des patientes atteintes du cancer du sein")
st.caption("Prototype de recherche, mémoire de master Statistique et Informatique Décisionnelle")
st.warning("Outil de recherche, non validé cliniquement. Il ne doit pas guider une décision de traitement. "
           "Aucune donnée saisie ici n'est enregistrée.")

with st.sidebar:
    st.header("Patiente")
    choix = st.radio("Critère à estimer", ["Survie globale", "Survie sans progression"],
                     help="La survie sans progression compte la récidive, la progression ou le décès.")
    critere = "os" if choix == "Survie globale" else "pfs"
    age = st.slider("Âge au diagnostic (ans)", 20, 90, 47)
    stade = st.selectbox("Stade global", STADES, index=1,
                         help="Regroupement du stade TNM : I-II, III ou IV.")
    grade = st.selectbox("Grade SBR", GRADES, index=1)
    sous_type = st.selectbox("Sous-type immunohistochimique", SOUS_TYPES,
                             help="Luminal (RO ou RP positif, HER2 négatif), triple négatif, ou HER2 positif.")
    if critere == "os" and stade == "III":
        st.info("Pour la survie globale, les stades I-II et III forment un seul groupe (effectifs trop faibles).")

x = encoder(critere, age, stade, grade, sous_type)
r = predire(M, critere, x)
G = M[critere]["groupes"][r["groupe"]]

col1, col2 = st.columns([1.1, 1])
with col1:
    st.subheader(f"{M[critere]['libelle']} estimée")
    a, b, c = st.columns(3)
    for colonne, mois, cle in ((a, 12, "a12"), (b, 24, "a24"), (c, 36, "a36")):
        v = r[cle]
        colonne.metric(f"À {mois} mois", f"{100*v['p']:.0f} %", help=f"IC 95 % : {100*v['bas']:.0f} à {100*v['haut']:.0f} %")
        colonne.caption(f"IC 95 % {100*v['bas']:.0f}-{100*v['haut']:.0f} %")
    nom = G["nom"]
    (st.success if r["groupe"] == 0 else st.warning if r["groupe"] == 1 else st.error)(
        f"{nom} : risque relatif ×{fr(r['risque_relatif'])} par rapport à la patiente moyenne de la cohorte.")
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for g, couleur in enumerate(COULEURS):
        gg = M[critere]["groupes"][g]
        ax.step(gg["km_t"], gg["km_s"], where="post", color=couleur, lw=1.2, alpha=0.9 if g == r["groupe"] else 0.35,
                label=f"{gg['nom']} (n={gg['n']}, observée)")
    ax.fill_between(r["mois"], r["bas"], r["haut"], color="#1F3A5F", alpha=0.18, lw=0)
    ax.plot(r["mois"], r["S"], color="#1F3A5F", lw=2.4, label="Cette patiente (estimée)")
    ax.set_xlim(0, 36); ax.set_ylim(0.2, 1.01); ax.set_xlabel("Mois depuis le diagnostic"); ax.set_ylabel("Probabilité")
    ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, fontsize=7, loc="lower left")
    st.pyplot(fig); plt.close(fig)
with col2:
    st.subheader("Facteurs qui expliquent l'estimation")
    noms = [n for n, _ in r["contributions"]][::-1]; mult = np.array([m for _, m in r["contributions"]])[::-1]
    fig2, ax2 = plt.subplots(figsize=(5.4, 3.8))
    ax2.barh(noms, np.log(mult), color=["#C0504D" if m > 1 else "#4F9D69" for m in mult])
    ax2.axvline(0, color="#777", lw=0.8); ax2.set_xlabel("Effet sur le risque (échelle log)")
    for i, m in enumerate(mult): ax2.text(np.log(m) + (0.02 if m >= 1 else -0.02), i, f"×{fr(m)}", va="center", ha="left" if m >= 1 else "right", fontsize=8)
    ax2.spines[["top", "right"]].set_visible(False); st.pyplot(fig2); plt.close(fig2)
    st.caption("Chaque barre compare la patiente à la patiente moyenne de la cohorte : ×1,5 signifie 50 % de risque en plus à autres facteurs égaux. "
               "Ces effets décrivent le modèle, pas des causes.")

st.subheader("Compte-rendu automatique")
texte = compte_rendu(M, critere, age, stade, grade, sous_type, r)
st.text_area("Texte généré", texte, height=210, label_visibility="collapsed")
st.download_button("Télécharger le compte-rendu (.txt)", texte, file_name="compte_rendu.txt")

with st.expander("Limites et fiabilité du modèle"):
    P = M[critere]
    st.markdown(
        f"- **Données** : {P['n']} patientes d'un seul hôpital, {P['evenements']} événements, suivi médian {P['suivi_median']:.0f} mois, "
        "environ un tiers de perdues de vue.\n"
        f"- **Discrimination** : C-index {P['c_interne'][0]:.2f} (IC {P['c_interne'][1]:.2f}-{P['c_interne'][2]:.2f}) en validation croisée, "
        f"{P['c_externe'][0]:.2f} (IC {P['c_externe'][1]:.2f}-{P['c_externe'][2]:.2f}) sur la cohorte externe METABRIC.\n"
        "- **Variables** : âge, stade, grade et sous-type seulement ; ni ganglions ni taille tumorale, donc moins précis que l'indice de Nottingham.\n"
        "- **Incertitude** : les intervalles viennent d'un bootstrap et sont larges ; ils ne couvrent pas les biais de recueil.\n"
        "- **Horizon** : au-delà de 36 mois, aucune estimation n'est fournie.")
st.caption("Version du prototype : novembre 2026.")
