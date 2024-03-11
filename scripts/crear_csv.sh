#!/bin/bash

# Verificar que se proporcionan dos directorios como argumentos
if [ $# -ne 2 ]; then
    echo "Uso: $0 directorio_origen directorio_destino"
    exit 1
fi

# Obtener el directorio origen y destino
directorio_origen="$1"
directorio_destino="$2"

# Verificar que los directorios existen
if [ ! -d "$directorio_origen" ] || [ ! -d "$directorio_destino" ]; then
    echo "Los directorios especificados no existen."
    exit 1
fi

# Iterar sobre todos los archivos .wav en el directorio de origen
for archivo in "$directorio_origen"/*.wav; do
    # Verificar que el elemento es un archivo
    if [ -f "$archivo" ]; then
        # Extraer el nombre del archivo sin extensión y sumarle 20
        nombre_archivo=$(basename "$archivo" .wav)
        nota=$(expr $nombre_archivo + 20)

        # Obtener el número de muestras del archivo wav
        muestras=$(soxi -s "$archivo")

        # Crear el archivo CSV en el directorio destino
        csv_file="$directorio_destino/${nombre_archivo}.csv"
        echo "start_time,end_time,instrument,note" > "$csv_file"
        echo "0,$muestras,1,$nota" >> "$csv_file"

        echo "Se ha creado el archivo $csv_file"
    fi
done

echo "Proceso completado."



