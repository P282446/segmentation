# -*- coding: utf-8 -*-
# Segmentation et evaluation globale avec reprise Grid5000.
# Seul resultat public : resultats_segmentation.csv.
# Les fichiers internes et sorties temporaires servent a la reprise.
# Les sorties historiques ne sont pas effacees automatiquement.
# Usage : python3 evaluation_segmentation.py SOURCE SORTIE --heures 12 --marge 10

import cv2
import numpy as np
import pandas as pd

import argparse
import subprocess
import time
import signal
import json
import sys
import os
import shutil
import fcntl

from pathlib import Path
from collections import deque
from tqdm import tqdm


# =====================================================================
# 2. PARAMETRES GENERAUX
# =====================================================================

PIPELINE_DIR = Path(__file__).resolve().parent

APP_FILE = PIPELINE_DIR / "app.py"

VIDEO_EXTENSIONS = {
    ".mp4",
    ".wmv",
    ".avi",
    ".mov",
    ".mkv"
}

DUREE_RESERVATION_HEURES = 12

MARGE_SECURITE_MINUTES = 10


# =====================================================================
# 3. COULEURS DES REGIONS ANATOMIQUES
# =====================================================================

# Couleurs BGR observees sur l'image segmentee de reference.

COLORS_BGR = {
    1: (97, 254, 0),      # Vert : Yeux
    2: (99, 0, 252),      # Rose : Nez et bouche
    3: (94, 254, 253),    # Jaune : Visage
    4: (0, 0, 250),       # Rouge : Corps
    5: (96, 0, 0)         # Bleu : Reste de la tete
}

ZONES = {
    1: "Yeux",
    2: "Bouche_Nez",
    3: "Visage",
    4: "Corps",
    5: "Reste_tete"
}


# =====================================================================
# 4. PARAMETRES DE VALIDATION DE LA SEGMENTATION
# =====================================================================

# Tolerance liee a la compression video
TOLERANCE_COULEUR = 25

# Nombre minimum de pixels pour considerer une region presente
MIN_PIXELS_ZONE = 20

# Surface segmentee minimale : 0.5 % de la frame
MIN_SURFACE_SEGMENTEE = 0.005

# Proportions minimales et maximales des regions
# relativement a la surface totale segmentee
PROPORTIONS = {
    1: (0.02, 0.16),   # Yeux
    2: (0.02, 0.20),   # Nez et bouche
    3: (0.05, 0.40),   # Visage
    4: (0.20, 0.85),   # Corps
    5: (0.01, 0.35)    # Reste de la tete
}

# Comparaisons des surfaces
COMPARAISONS_SURFACES = [
    (1, 3),
    (1, 4),
    (2, 3),
    (2, 4),
    (3, 4),
    (5, 4)
]

# Continuite geometrique
MIN_CONTINUITE = 0.70

# Tolerance spatiale
TOLERANCE_POSITION = 0.30

# Analyse temporelle
FENETRE_TEMPORELLE = 30

MIN_HISTORIQUE = 10

FACTEUR_VARIATION = 3.0

SEUIL_VARIATION_ABSOLUE = 0.05

# Activation ou desactivation des controles
ACTIVER_COMPARAISON_SURFACES = True
ACTIVER_CONTROLE_SPATIAL = True
ACTIVER_CONTROLE_GEOMETRIQUE = True
ACTIVER_CONTROLE_TEMPOREL = True


# =====================================================================
# 5. COLONNES DU FICHIER CSV FINAL
# =====================================================================

COLONNES_RESULTATS = [
    "video",
    "nombre_total_frames",
    "pct_frames_correctement_segmentees",
    "pct_frames_mal_segmentees",
    "duree_video_secondes",
    "duree_segmentation_secondes"
]


# =====================================================================
# 6. INTERVALLES DE COULEURS
# =====================================================================

INTERVALLES_COULEURS = {}

for code, couleur in COLORS_BGR.items():

    couleur = np.array(
        couleur,
        dtype=np.int16
    )

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


# =====================================================================
# 7. GESTION DES INTERRUPTIONS
# =====================================================================

class FinReservation(Exception):
    pass


ARRET_DEMANDE = False


def interruption(signum, frame):

    global ARRET_DEMANDE

    ARRET_DEMANDE = True


signal.signal(signal.SIGTERM, interruption)

signal.signal(signal.SIGINT, interruption)

signal.signal(signal.SIGHUP, interruption)


def verifier_temps(deadline):

    if ARRET_DEMANDE:

        raise FinReservation(
            "Interruption demandee."
        )

    if time.monotonic() >= deadline:

        raise FinReservation(
            "Fin du temps de calcul disponible."
        )


# =====================================================================
# 8. SAUVEGARDE ET CHARGEMENT DE L'ETAT
# =====================================================================

def sauvegarder_etat(etat, fichier_etat):

    fichier_temporaire = (
        fichier_etat.with_suffix(".tmp")
    )

    with open(
        fichier_temporaire,
        "w",
        encoding="utf-8"
    ) as fichier:

        json.dump(
            etat,
            fichier,
            indent=4,
            ensure_ascii=False
        )

        fichier.flush()

        os.fsync(fichier.fileno())

    os.replace(
        fichier_temporaire,
        fichier_etat
    )


def charger_etat(fichier_etat):

    if fichier_etat.exists():

        with open(
            fichier_etat,
            "r",
            encoding="utf-8"
        ) as fichier:

            return json.load(fichier)

    return {}


# =====================================================================
# 9. EXPORTATION DES RESULTATS
# =====================================================================

def exporter_resultats(etat, fichier_resultats):

    resultats = []

    for information in etat.values():

        if information.get("statut") == "terminee":

            resultats.append(
                information["resultat"]
            )

    df = pd.DataFrame(
        resultats,
        columns=COLONNES_RESULTATS
    )

    fichier_temporaire = (
        fichier_resultats.with_suffix(".tmp")
    )

    df.to_csv(
        fichier_temporaire,
        index=False,
        float_format="%.2f"
    )

    os.replace(
        fichier_temporaire,
        fichier_resultats
    )





# =====================================================================
# 10. INFORMATIONS SUR LA VIDEO ORIGINALE
# =====================================================================

def informations_video(video_path, deadline):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir : {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        nombre_frames = 0
        while True:
            verifier_temps(deadline)
            if not cap.grab():
                break
            nombre_frames += 1
    finally:
        cap.release()
    if not np.isfinite(fps) or fps <= 0 or not nombre_frames:
        raise RuntimeError("Video vide ou frequence invalide.")
    return nombre_frames, fps, nombre_frames / fps



# =====================================================================
# 11. EXTRACTION DES REGIONS ANATOMIQUES
# =====================================================================

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

        # Centre de gravite
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

        # Continuite geometrique
        continuite = 0.0

        if ACTIVER_CONTROLE_GEOMETRIQUE:

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

        # Rectangle englobant
        bbox = cv2.boundingRect(masque)

        regions[code] = {

            "surface": surface,

            "centre": centre,

            "continuite": continuite,

            "bbox": bbox
        }

    return regions


# =====================================================================
# 12. EVALUATION DE LA QUALITE D'UNE FRAME
# =====================================================================

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
            if surface_totale > 0
            else 0.0
        )

        for code, surface in surfaces.items()
    }

    # -----------------------------------------------------------------
    # CRITERE 1 : ABSENCE DE SEGMENTATION
    # -----------------------------------------------------------------

    if surface_totale == 0:

        erreurs.append(
            "Absence_segmentation"
        )

        return (
            False,
            erreurs,
            regions,
            proportions
        )

    if (
        surface_totale / nombre_pixels_frame
        < MIN_SURFACE_SEGMENTEE
    ):

        erreurs.append(
            "Surface_segmentee_insuffisante"
        )

    for code, surface in surfaces.items():

        if surface == 0:

            erreurs.append(
                f"Region_absente_{ZONES[code]}"
            )

    # -----------------------------------------------------------------
    # CRITERE 2 : PROPORTIONS ANATOMIQUES
    # -----------------------------------------------------------------

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

    # -----------------------------------------------------------------
    # CRITERE 3 : COMPARAISON DES SURFACES
    # -----------------------------------------------------------------

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

    # -----------------------------------------------------------------
    # CRITERE 4 : COHERENCE SPATIALE
    # -----------------------------------------------------------------

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

        # Le nez et la bouche doivent etre au-dessus du corps
        if (
            centre_bouche is not None
            and centre_corps is not None
        ):

            if centre_bouche[1] >= centre_corps[1]:

                erreurs.append(
                    "Position_bouche_anormale"
                )

        # Alignement horizontal des regions du visage
        if (
            regions[3]["bbox"] is not None
            and centre_visage is not None
        ):

            x, y, largeur, hauteur = regions[3]["bbox"]

            tolerance_x = (
                largeur * TOLERANCE_POSITION
            )

            if centre_yeux is not None:

                distance_x = abs(
                    centre_yeux[0] - centre_visage[0]
                )

                if distance_x > tolerance_x:

                    erreurs.append(
                        "Mauvais_alignement_yeux"
                    )

            if centre_bouche is not None:

                distance_x = abs(
                    centre_bouche[0] - centre_visage[0]
                )

                if distance_x > tolerance_x:

                    erreurs.append(
                        "Mauvais_alignement_bouche"
                    )

    # -----------------------------------------------------------------
    # CRITERE 5 : COHERENCE GEOMETRIQUE
    # -----------------------------------------------------------------

    if ACTIVER_CONTROLE_GEOMETRIQUE:

        for code, region in regions.items():

            if region["surface"] == 0:

                continue

            seuil = MIN_CONTINUITE

            # Les yeux peuvent former deux composantes distinctes
            if code == 1:

                seuil = 0.40

            if region["continuite"] < seuil:

                erreurs.append(
                    f"Fragmentation_{ZONES[code]}"
                )

    # -----------------------------------------------------------------
    # CRITERE 6 : COHERENCE TEMPORELLE
    # -----------------------------------------------------------------

    valide_structurellement = (
        len(erreurs) == 0
    )

    if (
        ACTIVER_CONTROLE_TEMPOREL
        and valide_structurellement
    ):

        for code, proportion in proportions.items():

            valeurs = historique[code]

            if len(valeurs) < MIN_HISTORIQUE:

                continue

            mediane = float(
                np.median(valeurs)
            )

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
                    f"Variation_temporelle_{ZONES[code]}"
                )

    # Mise a jour de l'historique temporel
    if valide_structurellement:

        for code, proportion in proportions.items():

            historique[code].append(
                proportion
            )

    correcte = len(erreurs) == 0

    return (
        correcte,
        erreurs,
        regions,
        proportions
    )


# =====================================================================
# 13. EVALUATION COMPLETE D'UNE VIDEO SEGMENTEE
# =====================================================================

def evaluer_video(video_path, nombre_total_frames, deadline):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir : {video_path}")
    nombre_frames = nombre_correctes = 0
    historique = {code: deque(maxlen=FENETRE_TEMPORELLE) for code in ZONES}
    try:
        while True:
            verifier_temps(deadline)
            ret, frame = cap.read()
            if not ret:
                break
            correcte, _, _, _ = evaluer_frame(frame, historique)
            nombre_frames += 1
            nombre_correctes += int(correcte)
    finally:
        cap.release()
    if nombre_frames == 0:
         raise RuntimeError("Aucune frame segmentee lisible.")

    if nombre_total_frames <= 0:
         raise RuntimeError("Nombre de frames originales invalide.")

    ecart = abs(nombre_frames - nombre_total_frames)
    ecart_relatif = ecart / nombre_total_frames

    # Tolerance maximale de 0,1 %
    if ecart_relatif > 0.001:
         raise RuntimeError(
             f"Ecart trop important : "
             f"originale={nombre_total_frames}, "
             f"segmentee={nombre_frames}, "
             f"ecart={ecart_relatif * 100:.4f}%"
            )

    return (
       nombre_frames,
       nombre_correctes,
       nombre_frames - nombre_correctes
       )


# =====================================================================
# 14. ARRET DU PROCESSUS DE SEGMENTATION
# =====================================================================

def arreter_processus(processus):

    if processus.poll() is not None:

        return

    try:

        os.killpg(
            processus.pid,
            signal.SIGTERM
        )

        processus.wait(timeout=5)

    except subprocess.TimeoutExpired:

        try:

            os.killpg(
                processus.pid,
                signal.SIGKILL
            )

        except ProcessLookupError:

            pass

        processus.wait()

    except ProcessLookupError:

        pass


# =====================================================================
# 15. SEGMENTATION ET CHRONOMETRAGE
# =====================================================================

def segmenter_video(video_path, output_dir, deadline):
    verifier_temps(deadline)
    commande = [sys.executable, "-u", str(APP_FILE), str(video_path), str(output_dir)]
    debut = time.perf_counter()
    processus = subprocess.Popen(commande, cwd=PIPELINE_DIR, start_new_session=True,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        while processus.poll() is None:
            verifier_temps(deadline)
            time.sleep(1)
        if processus.returncode != 0:
            raise RuntimeError(f"Echec de app.py : code {processus.returncode}")
    finally:
        arreter_processus(processus)
    return time.perf_counter() - debut



# =====================================================================
# 16. RECHERCHE DE LA VIDEO SEGMENTEE
# =====================================================================

def trouver_video_segmentee(
    video_path,
    output_dir
):

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
        f"Video segmentee introuvable ou ambigue : "
        f"{video_path.name}"
    )


# =====================================================================
# 17. TRAITEMENT COMPLET D'UNE VIDEO
# =====================================================================

def traiter_video(
    video_path,
    source_dir,
    output_dir,
    etat,
    fichier_etat,
    fichier_resultats,
    deadline
):

    identifiant = str(
        video_path.relative_to(source_dir)
    )

    dossier_video = (
        output_dir
        / "videos"
        / identifiant
    )

    dossier_segmentation = (
        dossier_video / "segmentation"
    )

    dossier_video.mkdir(
        parents=True,
        exist_ok=True
    )

    information = etat.setdefault(
        identifiant,
        {"statut": "en_attente"}
    )

    sauvegarder_etat(
        etat,
        fichier_etat
    )

    # -----------------------------------------------------------------
    # ETAPE 1 : INFORMATIONS SUR LA VIDEO ORIGINALE
    # -----------------------------------------------------------------

    if "nombre_total_frames" not in information:

        (
            nombre_frames,
            fps,
            duree_video

        ) = informations_video(
            video_path,
            deadline
        )

        information["nombre_total_frames"] = (
            nombre_frames
        )

        information["fps"] = fps

        information["duree_video_secondes"] = (
            duree_video
        )

        sauvegarder_etat(
            etat,
            fichier_etat
        )




    # -----------------------------------------------------------------
    # ETAPE 2 : VERIFICATION DE LA SEGMENTATION
    # -----------------------------------------------------------------

    segmentation_terminee = (
        information.get("statut") in {
            "segmentation_terminee",
            "evaluation_en_cours"
        }
    )

    if segmentation_terminee:

        video_segmentee = Path(
            information["video_segmentee"]
        )

        if not video_segmentee.is_file():

            segmentation_terminee = False

    # -----------------------------------------------------------------
    # ETAPE 3 : SEGMENTATION
    # -----------------------------------------------------------------

    if not segmentation_terminee:

        verifier_temps(deadline)

        information["statut"] = (
            "segmentation_en_cours"
        )

        sauvegarder_etat(
            etat,
            fichier_etat
        )

        # Supprimer uniquement les sorties partielles
        # du dossier dedie a cette segmentation.

        if dossier_segmentation.exists():

            shutil.rmtree(
                dossier_segmentation
            )

        dossier_segmentation.mkdir(
            parents=True,
            exist_ok=True
        )

        duree_segmentation = segmenter_video(
            video_path,
            dossier_segmentation,
            deadline
        )

        video_segmentee = trouver_video_segmentee(
            video_path,
            dossier_segmentation
        )

        information["video_segmentee"] = str(
            video_segmentee
        )

        information["duree_segmentation_secondes"] = (
            duree_segmentation
        )

        information["statut"] = (
            "segmentation_terminee"
        )

        sauvegarder_etat(
            etat,
            fichier_etat
        )

    # -----------------------------------------------------------------
    # ETAPE 4 : EVALUATION DE LA QUALITE
    # -----------------------------------------------------------------

    verifier_temps(deadline)

    information["statut"] = (
        "evaluation_en_cours"
    )

    sauvegarder_etat(
        etat,
        fichier_etat
    )


    (
        nombre_frames,
        nombre_correctes,
        nombre_incorrectes

    ) = evaluer_video(
        video_segmentee,
        information["nombre_total_frames"],
        deadline
    )

    # -----------------------------------------------------------------
    # ETAPE 5 : CALCUL DES STATISTIQUES
    # -----------------------------------------------------------------

    pct_correctes = (
        nombre_correctes / nombre_frames * 100
    )

    pct_incorrectes = (
        nombre_incorrectes / nombre_frames * 100
    )

    resultat = {

        "video": identifiant,

        "nombre_total_frames":
            nombre_frames,

        "pct_frames_correctement_segmentees":
            round(pct_correctes, 2),

        "pct_frames_mal_segmentees":
            round(pct_incorrectes, 2),

        "duree_video_secondes":
            round(
                information["duree_video_secondes"],
                2
            ),

        "duree_segmentation_secondes":
            round(
                information["duree_segmentation_secondes"],
                2
            )
    }

    # -----------------------------------------------------------------
    # ETAPE 6 : SAUVEGARDE DES RESULTATS
    # -----------------------------------------------------------------

    information["resultat"] = resultat

    information["statut"] = "terminee"

    information.pop("erreur", None)

    sauvegarder_etat(
        etat,
        fichier_etat
    )

    exporter_resultats(
        etat,
        fichier_resultats
    )

    # La sauvegarde est terminee : supprimer les sorties de cette video.
    try:
        shutil.rmtree(dossier_segmentation)
        if not any(dossier_video.iterdir()):
            dossier_video.rmdir()
    except OSError as erreur:
        print(f"Nettoyage incomplet : {erreur}", file=sys.stderr)







# =====================================================================
# 18. GESTION DU PIPELINE GLOBAL
# =====================================================================

def executer_pipeline(
    source_dir,
    output_dir,
    temps_disponible
):

    debut_execution = time.monotonic()

    deadline = (
        debut_execution + temps_disponible
    )

    fichier_etat = (
        output_dir / "etat_avancement.json"
    )

    fichier_resultats = (
        output_dir / "resultats_segmentation.csv"
    )


    # Chargement des traitements precedents
    etat = charger_etat(fichier_etat)

    # Verification du dossier source
    fichier_source = (
        output_dir / "source.txt"
    )

    if fichier_source.exists():

        ancienne_source = (
            fichier_source.read_text(
                encoding="utf-8"
            ).strip()
        )

        if ancienne_source != str(source_dir):

            raise ValueError(
                "Le dossier de sortie appartient "
                "a un autre dossier source."
            )

    else:

        if etat:

            raise RuntimeError(
                "Etat existant sans identification "
                "du dossier source."
            )

        fichier_source.write_text(
            str(source_dir),
            encoding="utf-8"
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

    identifiants = {

        str(video.relative_to(source_dir))

        for video in videos
    }

    if set(etat) - identifiants:

        raise RuntimeError(
            "Certaines videos deja enregistrees "
            "ont disparu du dossier source."
        )

    # Initialisation et verification des videos sources
    for video in videos:

        identifiant = str(
            video.relative_to(source_dir)
        )

        stat = video.stat()

        empreinte = {
            "taille": stat.st_size,
            "modification_ns": stat.st_mtime_ns
        }

        information = etat.setdefault(
            identifiant,
            {"statut": "en_attente"}
        )

        ancienne_empreinte = (
            information.get("source")
        )

        if (
            ancienne_empreinte is not None
            and ancienne_empreinte != empreinte
        ):

            raise RuntimeError(
                f"Video originale modifiee : {video}"
            )

        information["source"] = empreinte

    sauvegarder_etat(
        etat,
        fichier_etat
    )

    nombre_terminees = sum(

        information.get("statut") == "terminee"

        for information in etat.values()
    )

    exporter_resultats(
        etat,
        fichier_resultats
    )


    try:

        with tqdm(
            total=len(videos),
            initial=nombre_terminees,
            desc="Progression globale",
            unit="video",
            position=0,
            leave=True,
            dynamic_ncols=True
        ) as barre_globale:

            for video in videos:

                verifier_temps(deadline)

                identifiant = str(
                    video.relative_to(source_dir)
                )

                information = etat[identifiant]

                # Ignorer les videos entierement traitees
                if information.get("statut") == "terminee":

                    continue

                temps_restant = max(
                    0,
                    deadline - time.monotonic()
                )

                barre_globale.set_postfix(

                    video=video.name,

                    restant_h=round(
                        temps_restant / 3600,
                        2
                    )
                )

                try:

                    traiter_video(
                        video,
                        source_dir,
                        output_dir,
                        etat,
                        fichier_etat,
                        fichier_resultats,
                        deadline
                    )

                    barre_globale.update(1)

                except FinReservation:

                    raise

                except Exception as erreur:

                    tqdm.write(
                        f"\nERREUR : {video.name}"
                    )

                    tqdm.write(
                        str(erreur)
                    )

                    information["erreur"] = str(
                        erreur
                    )

                    sauvegarder_etat(
                        etat,
                        fichier_etat
                    )


    except FinReservation as erreur:

        print(f"\n{erreur}")

        print(
            "Arret du traitement. "
            "Les resultats sont conserves."
        )

    finally:

        sauvegarder_etat(
            etat,
            fichier_etat
        )

        exporter_resultats(
            etat,
            fichier_resultats
        )


        print(f"Resultats : {fichier_resultats}")


# =====================================================================
# 19. PROGRAMME PRINCIPAL
# =====================================================================

def main():

    parser = argparse.ArgumentParser(

        description=(
            "Segmentation automatique et evaluation "
            "avec reprise sur Grid5000"
        )
    )

    parser.add_argument(
        "source_dir",
        help="Dossier des videos originales"
    )

    parser.add_argument(
        "output_dir",
        help="Dossier des resultats"
    )

    parser.add_argument(
        "--heures",
        type=float,
        default=DUREE_RESERVATION_HEURES,
        help="Temps de calcul restant en heures"
    )

    parser.add_argument(
        "--marge",
        type=float,
        default=MARGE_SECURITE_MINUTES,
        help="Marge de securite en minutes"
    )

    args = parser.parse_args()

    source_dir = Path(
        args.source_dir
    ).resolve()

    output_dir = Path(
        args.output_dir
    ).resolve()

    # Verification des dossiers
    if not source_dir.is_dir():

        raise NotADirectoryError(
            source_dir
        )

    if not APP_FILE.is_file():

        raise FileNotFoundError(
            f"Programme introuvable : {APP_FILE}"
        )

    if source_dir.is_relative_to(output_dir):

        raise ValueError(
            "Le dossier de sortie ne doit pas "
            "contenir le dossier source."
        )

    if (
        not np.isfinite(args.heures)
        or not np.isfinite(args.marge)
        or args.heures <= 0
        or args.marge < 0
    ):

        raise ValueError(
            "Parametres de duree invalides."
        )

    # Calcul du temps disponible
    temps_disponible = (
        args.heures * 3600
        - args.marge * 60
    )

    if temps_disponible <= 0:

        raise ValueError(
            "Temps de calcul disponible insuffisant."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Empecher deux executions simultanees
    # sur le meme dossier de resultats.
    fichier_verrou = (
        output_dir / "execution.lock"
    )

    with open(fichier_verrou, "a+") as verrou:

        try:

            fcntl.flock(
                verrou,
                fcntl.LOCK_EX | fcntl.LOCK_NB
            )

        except BlockingIOError:

            raise RuntimeError(
                "Une autre execution utilise "
                "deja ce dossier de resultats."
            )

        try:

            executer_pipeline(
                source_dir,
                output_dir,
                temps_disponible
            )

        finally:

            fcntl.flock(
                verrou,
                fcntl.LOCK_UN
            )


# =====================================================================
# 20. EXECUTION DU PROGRAMME
# =====================================================================

if __name__ == "__main__":

    main()
