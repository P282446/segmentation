
import cv2
import numpy as np
import pandas as pd
import subprocess
import argparse
import time
import sys
import csv

from pathlib import Path
from collections import deque, Counter


# ============================================================
# 1. PARAMETRES GENERAUX
# ============================================================

PIPELINE_DIR = Path(__file__).resolve().parent

APP_FILE = PIPELINE_DIR / "app.py"

VIDEO_EXTENSIONS = {
    ".mp4",
    ".wmv",
    ".avi",
    ".mov",
    ".mkv"
}

# Couleurs BGR observees sur la video segmentee
COLORS_BGR = {
    1: (97, 254, 0),     # Vert : Yeux
    2: (99, 0, 252),     # Rose : Nez et bouche
    3: (94, 254, 253),   # Jaune : Visage
    4: (0, 0, 250),     # Rouge : Corps
    5: (96, 0, 0)       # Bleu : Reste de la tete
}

ZONES = {
    1: "Yeux",
    2: "Bouche_Nez",
    3: "Visage",
    4: "Corps",
    5: "Reste_tete"
}


# ============================================================
# 2. PARAMETRES DE VALIDATION
# ============================================================

# Tolerance pour tenir compte de la compression video
TOLERANCE_COULEUR = 25

# Nombre minimum de pixels par region
MIN_PIXELS_ZONE = 20

# Surface segmentee minimale : 0.5 % de la frame
MIN_SURFACE_SEGMENTEE = 0.005

# Proportions minimales et maximales des regions
# par rapport a la surface totale segmentee
PROPORTIONS = {
    1: (0.02, 0.16),  # Yeux
    2: (0.02, 0.20),  # Nez et bouche
    3: (0.05, 0.40),  # Visage
    4: (0.20, 0.85),  # Corps
    5: (0.01, 0.35)   # Reste de la tete
}

# Comparaison des surfaces anatomiques
COMPARAISONS_SURFACES = [
    (1, 3),  # Yeux < Visage
    (1, 4),  # Yeux < Corps
    (2, 3),  # Bouche_Nez < Visage
    (2, 4),  # Bouche_Nez < Corps
    (3, 4),  # Visage < Corps
    (5, 4)   # Reste_tete < Corps
]

# Proportion minimale de la zone devant appartenir
# a sa plus grande composante connexe
MIN_CONTINUITE = 0.70

# Tolerance spatiale horizontale
TOLERANCE_POSITION = 0.30

# Analyse temporelle
FENETRE_TEMPORELLE = 30

MIN_HISTORIQUE = 10

FACTEUR_VARIATION = 3.0

SEUIL_VARIATION_ABSOLUE = 0.05

# Activation des differents controles
ACTIVER_COMPARAISON_SURFACES = True
ACTIVER_CONTROLE_SPATIAL = True
ACTIVER_CONTROLE_GEOMETRIQUE = True
ACTIVER_CONTROLE_TEMPOREL = True


# ============================================================
# 3. CONSTRUCTION DES INTERVALLES DE COULEURS
# ============================================================

INTERVALLES_COULEURS = {}

for code, couleur in COLORS_BGR.items():

    couleur = np.array(couleur, dtype=np.int16)

    borne_inf = np.clip(
        couleur - TOLERANCE_COULEUR,
        0,
        255
    ).astype(np.uint8)

    borne_sup = np.clip(
        couleur + TOLERANCE_COULEUR,
        0,
        255
    ).astype(np.uint8)

    INTERVALLES_COULEURS[code] = (
        borne_inf,
        borne_sup
    )


# ============================================================
# 4. INFORMATIONS SUR LA VIDEO ORIGINALE
# ============================================================

def informations_video(video_path):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():

        raise RuntimeError(
            f"Impossible d'ouvrir : {video_path}"
        )

    try:

        fps = cap.get(cv2.CAP_PROP_FPS)

        nombre_frames = 0

        # Comptage des frames effectivement lisibles
        while cap.grab():

            nombre_frames += 1

        if fps <= 0 or nombre_frames == 0:

            raise RuntimeError(
                "Nombre de frames ou FPS invalide."
            )

        duree_video = nombre_frames / fps

        return nombre_frames, fps, duree_video

    finally:

        cap.release()


# ============================================================
# 5. EXTRACTION DES CARACTERISTIQUES DES REGIONS
# ============================================================

def extraire_regions(frame):

    regions = {}

    for code, bornes in INTERVALLES_COULEURS.items():

        borne_inf, borne_sup = bornes

        masque = cv2.inRange(
            frame,
            borne_inf,
            borne_sup
        )

        surface = cv2.countNonZero(masque)

        regions[code] = {
            "surface": 0,
            "centre": None,
            "continuite": 0.0,
            "bbox": None
        }

        if surface < MIN_PIXELS_ZONE:

            continue

        # Calcul du centre de gravite
        moments = cv2.moments(
            masque,
            binaryImage=True
        )

        if moments["m00"] > 0:

            cx = moments["m10"] / moments["m00"]

            cy = moments["m01"] / moments["m00"]

            centre = (cx, cy)

        else:

            centre = None

        # Recherche des composantes connexes
        nombre, labels, stats, centroids = (
            cv2.connectedComponentsWithStats(
                masque,
                connectivity=8
            )
        )

        if nombre > 1:

            plus_grande_surface = int(
                np.max(
                    stats[1:, cv2.CC_STAT_AREA]
                )
            )

            continuite = (
                plus_grande_surface / surface
            )

        else:

            continuite = 0.0

        # Rectangle englobant la region
        x, y, largeur, hauteur = cv2.boundingRect(
            masque
        )

        regions[code] = {
            "surface": surface,
            "centre": centre,
            "continuite": continuite,
            "bbox": (x, y, largeur, hauteur)
        }

    return regions


# ============================================================
# 6. CONTROLE DE LA QUALITE D'UNE FRAME
# ============================================================

def evaluer_frame(frame, historique):

    regions = extraire_regions(frame)

    erreurs = []

    surfaces = {
        code: region["surface"]
        for code, region in regions.items()
    }

    surface_totale = sum(surfaces.values())

    nombre_pixels_frame = (
        frame.shape[0] * frame.shape[1]
    )

    proportions = {
        code: (
            surface / surface_totale
            if surface_totale > 0 else 0.0
        )
        for code, surface in surfaces.items()
    }

    # --------------------------------------------------------
    # CRITERE 1 : Absence de segmentation
    # --------------------------------------------------------

    if surface_totale == 0:

        erreurs.append("Absence_segmentation")

        return (
            False,
            erreurs,
            regions,
            proportions
        )

    # Surface totale insuffisante
    if (
        surface_totale / nombre_pixels_frame
        < MIN_SURFACE_SEGMENTEE
    ):

        erreurs.append(
            "Surface_segmentee_insuffisante"
        )

    # Verification de la presence des cinq regions
    for code, surface in surfaces.items():

        if surface == 0:

            erreurs.append(
                f"Region_absente_{ZONES[code]}"
            )

    # --------------------------------------------------------
    # CRITERE 2 : Proportions anatomiques
    # --------------------------------------------------------

    for code, proportion in proportions.items():

        minimum, maximum = PROPORTIONS[code]

        if proportion < minimum:

            erreurs.append(
                f"Surface_insuffisante_{ZONES[code]}"
            )

        elif proportion > maximum:

            erreurs.append(
                f"Surface_excessive_{ZONES[code]}"
            )

    # --------------------------------------------------------
    # CRITERE 3 : Comparaison entre les surfaces
    # --------------------------------------------------------

    if ACTIVER_COMPARAISON_SURFACES:

        for petite_zone, grande_zone in (
            COMPARAISONS_SURFACES
        ):

            if (
                surfaces[petite_zone] > 0
                and surfaces[grande_zone] > 0
                and surfaces[petite_zone]
                >= surfaces[grande_zone]
            ):

                erreurs.append(
                    f"{ZONES[petite_zone]}"
                    f"_superieur_ou_egal_"
                    f"{ZONES[grande_zone]}"
                )

    # --------------------------------------------------------
    # CRITERE 4 : Coherence spatiale
    # --------------------------------------------------------

    if ACTIVER_CONTROLE_SPATIAL:

        centre_yeux = regions[1]["centre"]

        centre_bouche = regions[2]["centre"]

        centre_visage = regions[3]["centre"]

        centre_corps = regions[4]["centre"]

        # Les yeux doivent etre au-dessus de la bouche
        if (
            centre_yeux is not None
            and centre_bouche is not None
        ):

            if centre_yeux[1] >= centre_bouche[1]:

                erreurs.append(
                    "Yeux_en_dessous_de_la_bouche"
                )

        # Le corps doit etre en dessous du visage
        if (
            centre_corps is not None
            and centre_visage is not None
        ):

            if centre_corps[1] <= centre_visage[1]:

                erreurs.append(
                    "Corps_au_dessus_du_visage"
                )

        # Les yeux doivent etre au-dessus du corps
        if (
            centre_yeux is not None
            and centre_corps is not None
        ):

            if centre_yeux[1] >= centre_corps[1]:

                erreurs.append(
                    "Position_yeux_anormale"
                )

        # Le nez et la bouche doivent etre
        # au-dessus du corps
        if (
            centre_bouche is not None
            and centre_corps is not None
        ):

            if centre_bouche[1] >= centre_corps[1]:

                erreurs.append(
                    "Position_bouche_anormale"
                )

        # Alignement horizontal des zones du visage
        if regions[3]["bbox"] is not None:

            x, y, largeur_visage, hauteur_visage = (
                regions[3]["bbox"]
            )

            tolerance_x = (
                largeur_visage
                * TOLERANCE_POSITION
            )

            if (
                centre_yeux is not None
                and centre_visage is not None
            ):

                distance_x = abs(
                    centre_yeux[0]
                    - centre_visage[0]
                )

                if distance_x > tolerance_x:

                    erreurs.append(
                        "Mauvais_alignement_yeux"
                    )

            if (
                centre_bouche is not None
                and centre_visage is not None
            ):

                distance_x = abs(
                    centre_bouche[0]
                    - centre_visage[0]
                )

                if distance_x > tolerance_x:

                    erreurs.append(
                        "Mauvais_alignement_bouche"
                    )

    # --------------------------------------------------------
    # CRITERE 5 : Coherence geometrique
    # --------------------------------------------------------

    if ACTIVER_CONTROLE_GEOMETRIQUE:

        for code, region in regions.items():

            if region["surface"] == 0:

                continue

            seuil_continuite = MIN_CONTINUITE

            # Les yeux peuvent former deux composantes
            # connexes anatomiquement distinctes
            if code == 1:

                seuil_continuite = 0.40

            if region["continuite"] < seuil_continuite:

                erreurs.append(
                    f"Fragmentation_{ZONES[code]}"
                )

    # --------------------------------------------------------
    # CRITERE 6 : Coherence temporelle
    # --------------------------------------------------------

    # On compare uniquement les frames ayant passe
    # les controles precedents.

    valide_structurellement = len(erreurs) == 0

    if (
        ACTIVER_CONTROLE_TEMPOREL
        and valide_structurellement
    ):

        for code, proportion in proportions.items():

            valeurs = historique[code]

            if len(valeurs) < MIN_HISTORIQUE:

                continue

            mediane = float(np.median(valeurs))

            if mediane <= 0:

                continue

            rapport = proportion / mediane

            variation_absolue = abs(
                proportion - mediane
            )

            variation_anormale = (
                rapport > FACTEUR_VARIATION
                or rapport < 1 / FACTEUR_VARIATION
            )

            if (
                variation_anormale
                and variation_absolue
                > SEUIL_VARIATION_ABSOLUE
            ):

                erreurs.append(
                    f"Variation_temporelle_"
                    f"{ZONES[code]}"
                )

    # Actualisation de l'historique avec les frames
    # structurellement valides, afin de permettre
    # l'adaptation aux changements de position durables.

    if valide_structurellement:

        for code, proportion in proportions.items():

            historique[code].append(proportion)

    # --------------------------------------------------------
    # RESULTAT FINAL
    # --------------------------------------------------------

    correcte = len(erreurs) == 0

    return (
        correcte,
        erreurs,
        regions,
        proportions
    )


# ============================================================
# 7. EVALUATION DE LA VIDEO SEGMENTEE
# ============================================================

def evaluer_video(video_path, fichier_detail):

    print("\n" + "=" * 65)

    print(f"EVALUATION : {video_path.name}")

    print("=" * 65)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():

        raise RuntimeError(
            f"Impossible d'ouvrir : {video_path}"
        )

    nombre_frames = 0

    nombre_correctes = 0

    nombre_incorrectes = 0

    raisons_erreurs = Counter()

    historique = {
        code: deque(maxlen=FENETRE_TEMPORELLE)
        for code in ZONES
    }

    colonnes = [
        "Frame",
        "qualite_segmentation",
        "raisons"
    ]

    for zone in ZONES.values():

        colonnes.extend([
            f"pixels_{zone}",
            f"pct_{zone}",
            f"continuite_{zone}"
        ])

    try:

        with open(
            fichier_detail,
            "w",
            newline="",
            encoding="utf-8"
        ) as fichier:

            writer = csv.DictWriter(
                fichier,
                fieldnames=colonnes
            )

            writer.writeheader()

            while True:

                ret, frame = cap.read()

                if not ret:

                    break

                (
                    correcte,
                    erreurs,
                    regions,
                    proportions
                ) = evaluer_frame(
                    frame,
                    historique
                )

                if correcte:

                    nombre_correctes += 1

                else:

                    nombre_incorrectes += 1

                    raisons_erreurs.update(erreurs)

                ligne = {
                    "Frame": nombre_frames,
                    "qualite_segmentation": (
                        "Correcte"
                        if correcte
                        else "Incorrecte"
                    ),
                    "raisons": (
                        "OK"
                        if correcte
                        else " | ".join(erreurs)
                    )
                }

                for code, zone in ZONES.items():

                    ligne[f"pixels_{zone}"] = (
                        regions[code]["surface"]
                    )

                    ligne[f"pct_{zone}"] = round(
                        proportions[code] * 100,
                        2
                    )

                    ligne[f"continuite_{zone}"] = round(
                        regions[code]["continuite"]
                        * 100,
                        2
                    )

                writer.writerow(ligne)

                nombre_frames += 1

                if nombre_frames % 1000 == 0:

                    print(
                        f"\rFrames analysees : "
                        f"{nombre_frames}",
                        end="",
                        flush=True
                    )

    finally:

        cap.release()

    if nombre_frames == 0:

        raise RuntimeError(
            "Aucune frame lisible."
        )

    print("\n")

    print(f"Total frames : {nombre_frames}")

    print(f"Frames correctes : {nombre_correctes}")

    print(f"Frames incorrectes : {nombre_incorrectes}")

    print("\nPrincipales anomalies detectees :")

    for raison, nombre in raisons_erreurs.most_common():

        print(f"  {raison} : {nombre}")

    return (
        nombre_frames,
        nombre_correctes,
        nombre_incorrectes
    )


# ============================================================
# 8. EXECUTION ET CHRONOMETRAGE DE LA SEGMENTATION
# ============================================================

def segmenter_video(video_path, output_dir):

    commande = [
        sys.executable,
        str(APP_FILE),
        str(video_path),
        str(output_dir)
    ]

    print("\n" + "=" * 65)

    print(f"SEGMENTATION : {video_path.name}")

    print("=" * 65)

    debut = time.perf_counter()

    subprocess.run(
        commande,
        cwd=PIPELINE_DIR,
        check=True
    )

    fin = time.perf_counter()

    duree_segmentation = fin - debut

    print(
        f"\nDuree de segmentation : "
        f"{duree_segmentation:.2f} secondes"
    )

    return duree_segmentation


# ============================================================
# 9. RECHERCHE DE LA VIDEO SEGMENTEE
# ============================================================

def trouver_video_segmentee(video_path, output_dir):

    fichier_attendu = (
        output_dir / f"seg_{video_path.name}"
    )

    if fichier_attendu.is_file():

        return fichier_attendu

    fichiers = [

        fichier

        for fichier in output_dir.glob(
            f"seg_{video_path.stem}.*"
        )

        if fichier.is_file()

        and fichier.suffix.lower()
        in VIDEO_EXTENSIONS
    ]

    if len(fichiers) == 1:

        return fichiers[0]

    raise FileNotFoundError(
        f"Video segmentee introuvable "
        f"ou ambigue : {video_path.name}"
    )


# ============================================================
# 10. ANALYSE COMPLETE D'UNE VIDEO
# ============================================================

def analyser_video(
    video_path,
    output_dir,
    source_dir
):

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Informations sur la video originale
    (
        nombre_frames_originales,
        fps,
        duree_video
    ) = informations_video(video_path)

    print(
        f"\nNombre de frames : "
        f"{nombre_frames_originales}"
    )

    print(f"FPS : {fps:.2f}")

    print(
        f"Duree video : "
        f"{duree_video:.2f} secondes"
    )

    # Segmentation
    duree_segmentation = segmenter_video(
        video_path,
        output_dir
    )

    # Recherche de la video segmentee
    video_segmentee = trouver_video_segmentee(
        video_path,
        output_dir
    )

    # Fichier de controle frame par frame
    fichier_detail = (
        output_dir
        / f"qualite_{video_path.stem}.csv"
    )

    # Evaluation de la segmentation
    (
        nombre_frames_segmentees,
        nombre_correctes,
        nombre_incorrectes
    ) = evaluer_video(
        video_segmentee,
        fichier_detail
    )

    # Verification du nombre de frames
    if nombre_frames_segmentees != nombre_frames_originales:

        raise RuntimeError(
            f"Nombre de frames different : "
            f"originale={nombre_frames_originales}, "
            f"segmentee={nombre_frames_segmentees}"
        )

    # Calcul des pourcentages
    pct_correctes = (
        nombre_correctes
        / nombre_frames_originales
        * 100
    )

    pct_incorrectes = (
        nombre_incorrectes
        / nombre_frames_originales
        * 100
    )

    # Construction du resultat final
    resultat = {

        "video": str(
            video_path.relative_to(source_dir)
        ),

        "nombre_total_frames":
            nombre_frames_originales,

        "pct_frames_correctement_segmentees":
            round(pct_correctes, 2),

        "pct_frames_mal_segmentees":
            round(pct_incorrectes, 2),

        "duree_video_secondes":
            round(duree_video, 2),

        "duree_segmentation_secondes":
            round(duree_segmentation, 2)
    }

    return resultat


# ============================================================
# 11. PROGRAMME PRINCIPAL
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Segmentation automatique et evaluation "
            "de la qualite de segmentation"
        )
    )

    parser.add_argument(
        "source_dir",
        help="Dossier contenant les videos originales"
    )

    parser.add_argument(
        "output_dir",
        help="Dossier de sortie des resultats"
    )

    args = parser.parse_args()

    source_dir = Path(
        args.source_dir
    ).resolve()

    output_dir = Path(
        args.output_dir
    ).resolve()

    if not source_dir.is_dir():

        raise NotADirectoryError(source_dir)

    if not APP_FILE.is_file():

        raise FileNotFoundError(APP_FILE)

    if (
        source_dir == output_dir
        or source_dir.is_relative_to(output_dir)
    ):

        raise ValueError(
            "Le dossier de sortie ne doit pas "
            "contenir le dossier source."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    fichier_resultats = (
        output_dir
        / "resultats_segmentation.csv"
    )

    fichier_erreurs = (
        output_dir
        / "erreurs_segmentation.csv"
    )

    # Evite d'ecraser les resultats d'une ancienne execution
    if (
        fichier_resultats.exists()
        or fichier_erreurs.exists()
    ):

        raise FileExistsError(
            "Des resultats existent deja. "
            "Choisir un nouveau dossier de sortie."
        )

    # Recherche recursive des videos
    videos = sorted([

        fichier

        for fichier in source_dir.rglob("*")

        if fichier.is_file()

        and fichier.suffix.lower()
        in VIDEO_EXTENSIONS

        and not fichier.name.startswith("seg_")

        and not fichier.is_relative_to(output_dir)
    ])

    if not videos:

        print("Aucune video trouvee.")

        return

    print("\n" + "=" * 65)

    print("PIPELINE DE SEGMENTATION ET EVALUATION")

    print("=" * 65)

    print(
        f"Nombre de videos trouvees : "
        f"{len(videos)}"
    )

    resultats = []

    erreurs = []

    debut_pipeline = time.perf_counter()

    for i, video in enumerate(videos, start=1):

        print("\n" + "#" * 65)

        print(f"VIDEO {i}/{len(videos)}")

        print(f"Nom : {video.name}")

        print("#" * 65)

        try:

            # Un dossier de sortie par video
            dossier_video = (
                output_dir
                / video.parent.relative_to(source_dir)
                / video.name
            )

            if (
                dossier_video.exists()
                and any(dossier_video.iterdir())
            ):

                raise FileExistsError(
                    f"Dossier deja utilise : "
                    f"{dossier_video}"
                )

            resultat = analyser_video(
                video,
                dossier_video,
                source_dir
            )

            resultats.append(resultat)

            # Sauvegarde apres chaque video
            pd.DataFrame(resultats).to_csv(
                fichier_resultats,
                index=False,
                float_format="%.2f"
            )

            print("\nRESULTATS DE LA VIDEO")

            for cle, valeur in resultat.items():

                print(f"{cle} : {valeur}")

        except Exception as e:

            print(f"\nERREUR : {video.name}")

            print(str(e))

            erreurs.append({
                "video": str(video),
                "erreur": str(e)
            })

            pd.DataFrame(erreurs).to_csv(
                fichier_erreurs,
                index=False
            )

    fin_pipeline = time.perf_counter()

    duree_pipeline = (
        fin_pipeline - debut_pipeline
    )

    print("\n" + "=" * 65)

    print("TRAITEMENT TERMINE")

    print("=" * 65)

    print(
        f"Nombre de videos traitees : "
        f"{len(resultats)}"
    )

    print(
        f"Nombre de videos en echec : "
        f"{len(erreurs)}"
    )

    print(
        f"Duree totale du pipeline : "
        f"{duree_pipeline:.2f} secondes"
    )

    if resultats:

        print(
            f"Fichier des resultats : "
            f"{fichier_resultats}"
        )

    if erreurs:

        print(
            f"Fichier des erreurs : "
            f"{fichier_erreurs}"
        )


# ============================================================
# 12. EXECUTION
# ============================================================

if __name__ == "__main__":

    main()
