#!/bin/bash

# Dossier à parcourir (à adapter ou passer en argument)
SOURCE_DIR="${1:-.}"

# Dossier de sortie (2e argument, optionnel).
# Si non fourni, chaque vidéo est traitée dans son propre dossier source
OUTPUT_DIR="${2:-}"

# Extensions de vidéos à traiter
VIDEO_EXTENSIONS=("mp4" "wmv" "avi" "mov" "mkv")

# Construction du motif find pour les extensions
FIND_ARGS=()
for ext in "${VIDEO_EXTENSIONS[@]}"; do
    FIND_ARGS+=(-o -iname "*.${ext}")
done

# Retire le premier -o superflu
FIND_ARGS=("${FIND_ARGS[@]:1}")

# Si un dossier de sortie est précisé, on le crée
if [ -n "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

# Parcours récursif du dossier
find "$SOURCE_DIR" -type f \( "${FIND_ARGS[@]}" \) | while read -r video_path; do

    # Si OUTPUT_DIR est précisé, on l'utilise pour toutes les vidéos.
    # Sinon, comportement d'origine : sortie dans le dossier de la vidéo elle-même.
    if [ -n "$OUTPUT_DIR" ]; then
        target_dir="$OUTPUT_DIR"

        # Vérifie si cette vidéo a déjà été traitée
        video_name=$(basename "$video_path")

        if find "$target_dir" -maxdepth 1 -type f -name "seg_$video_name" | grep -q .; then
            echo "⏭️ Vidéo déjà traitée : $video_path"
            continue
        fi

    else
        target_dir=$(dirname "$video_path")
    fi

    echo "=================================================="
    echo "Traitement de : $video_path"
    echo "Sortie dans   : $target_dir"
    echo "=================================================="

    python3 app.py "$video_path" "$target_dir"

    if [ $? -eq 0 ]; then
        echo "✅ Terminé : $video_path"
    else
        echo "❌ Échec pour : $video_path"
    fi

done
