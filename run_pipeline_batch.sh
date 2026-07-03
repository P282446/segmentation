#!/bin/bash

# Dossier à parcourir (à adapter ou passer en argument)
SOURCE_DIR="${1:-.}"

# Extensions de vidéos à traiter (ajuste selon tes besoins)
VIDEO_EXTENSIONS=("mp4" "wmv" "avi" "mov" "mkv")

# Construction du motif find pour les extensions
FIND_ARGS=()
for ext in "${VIDEO_EXTENSIONS[@]}"; do
    FIND_ARGS+=(-o -iname "*.${ext}")
done
# Retire le premier -o superflu
FIND_ARGS=("${FIND_ARGS[@]:1}")

# Parcours récursif du dossier
find "$SOURCE_DIR" -type f \( "${FIND_ARGS[@]}" \) | while read -r video_path; do
    output_dir=$(dirname "$video_path")
    echo "=================================================="
    echo "Traitement de : $video_path"
    echo "Sortie dans   : $output_dir"
    echo "=================================================="

    python app.py "$video_path" "$output_dir"

    if [ $? -eq 0 ]; then
        echo "✅ Terminé : $video_path"
    else
        echo "❌ Échec pour : $video_path"
    fi
done
