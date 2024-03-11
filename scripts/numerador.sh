#!/bin/bash

# Verificar si se proporciona un directorio como argumento
if [ $# -ne 1 ]; then
    echo "Uso: $0 directorio"
    exit 1
fi

# Verificar si el directorio existe
if [ ! -d "$1" ]; then
    echo "El directorio '$1' no existe."
    exit 1
fi

# Cambiar al directorio especificado
cd "$1" || exit

# Enumerar los archivos por orden alfabético y renombrarlos
echo "Archivos en $1:"
i=1
for archivo in *; do
    extension="${archivo##*.}"
    nuevo_nombre="$(printf "%02d" $i).$extension"
    mv "$archivo" "$nuevo_nombre"
    echo "Renombrado $archivo a $nuevo_nombre"
    i=$((i + 1))
done

