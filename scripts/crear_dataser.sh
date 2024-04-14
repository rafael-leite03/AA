#!/bin/bash

# Carpeta de origen
carpeta_origen="/home/pajon/Descargas/3685367/TinySOL"

# Carpeta de destino
carpeta_destino="/home/pajon/Escritorio/Programacion/3º_2c/AA/practica/dataset_aux"

# Función para extraer la nota del nombre del archivo
extraer_nota() {
    # Utilizamos una expresión regular para extraer la nota
    if echo "$1" | grep -qE '([A-G][#b]?[0-9])'; then
        echo "$(basename "$1" | grep -oE '([A-G][#b]?[0-9])')"
    fi
}

# Recorrer todas las subcarpetas de la carpeta de origen
find "$carpeta_origen" -type f -iname "*.wav" | while read -r archivo; do
    # Obtener la nota del archivo
    nota=$(extraer_nota "$archivo")
    if [ -n "$nota" ]; then
        # Carpeta de destino para esta nota
        carpeta_nota="$carpeta_destino/$nota"
        # Crear la carpeta de destino si no existe
        mkdir -p "$carpeta_nota"
        # Copiar el archivo a la carpeta de destino si no existe previamente
        cp --no-clobber "$archivo" "$carpeta_nota"
    else
        echo "No se pudo extraer la nota para el archivo: $archivo"
    fi
done

echo "¡Proceso completado!"
