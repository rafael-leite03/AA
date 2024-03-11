#!/bin/bash

# Verificar que se proporciona un directorio como argumento
if [ $# -ne 1 ]; then
    echo "Uso: $0 directorio"
    exit 1
fi

# Cambiar al directorio especificado
cd "$1" || exit 1

# Iterar sobre todos los archivos en el directorio
for archivo in *; do
    # Verificar que el elemento es un archivo
    if [ -f "$archivo" ]; then
        # Extraer el nombre del archivo (sin extensión)
        nombre_sin_extension=$(basename "$archivo" | sed 's/\(.*\)\..*/\1/')
        
        # Obtener los dos últimos caracteres del nombre del archivo
        ultimos_caracteres=$(echo "$nombre_sin_extension" | rev | cut -c -2 | rev)

        # Cambiar el nombre del archivo
        nuevo_nombre="$ultimos_caracteres.${archivo##*.}"  # Conservar la extensión original
        mv "$archivo" "$nuevo_nombre"
        echo "Se ha cambiado el nombre de $archivo a $nuevo_nombre"
    fi
done

