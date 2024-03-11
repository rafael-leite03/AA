#!/bin/bash

# Verificar que se proporciona un directorio como argumento
if [ $# -ne 1 ]; then
    echo "Uso: $0 directorio"
    exit 1
fi

# Cambiar al directorio especificado
cd "$1" || exit 1

# Iterar sobre todos los archivos .wav en el directorio
for archivo in *.wav; do
    # Verificar que el elemento es un archivo
    if [ -f "$archivo" ]; then
        # Crear un nombre de archivo temporal para el archivo de salida
        archivo_temporal="temp_$archivo"

        # Convertir el archivo a mono y guardar en el archivo temporal
        sox "$archivo" -c 1 "$archivo_temporal" && echo "Se ha convertido $archivo a mono"

        # Reemplazar el archivo original con el archivo temporal
        mv "$archivo_temporal" "$archivo"
    fi
done

echo "Proceso completado."

