using WAV
using Flux
using SampledSignals
using CSV
using DataFrames

muestras_input=65536#potencia de 2 mayor o igual que cuatro

#Saca en un array los datos de .wav
function gen_input(ruta_archivo::AbstractString)
    # Leer el archivo .wav
    audio, sample_rate = WAV.wavread(ruta_archivo)
    
    # Normalizar la señal de audio
    audio = audio / maximum(abs.(audio))

    # Crear el input para la red neuronal (usando un arreglo unidimensional)
    input = reshape(audio, 1, :)
    return input
end

function obtener_info_wav(archivo)
    # Leer el archivo WAV
    audio, samplerate = wavread(archivo)
    
    # Obtener información del archivo WAV
    canales = size(audio, 2)
    frecuencia_muestreo = samplerate
    tasa_bits_muestra = eltype(audio) == Float32 ? 32 : 16  # Tasa de bits por muestra (bits)
    tamano = size(audio, 1) 
    # Calcular la tasa de bits de codificación (kbps)
    tasa_bits_codificacion = tasa_bits_muestra * frecuencia_muestreo * canales / 1000
    
    return canales, frecuencia_muestreo, tasa_bits_codificacion, tamano
end

#Convierte el array de datos del .wav en una matriz con un historico de muestras_input valores
function convertir_array(array::Matrix{T}, veces::Int) where T
    # Calcular el tamaño del nuevo array
    n_filas = veces
    n_columnas = (length(array))/(veces)
    n_columnas=round(Int,n_columnas)
    if (length(array))-veces/4>(n_columnas-1)*(veces*3/4)+(veces*3/4)
        n_columnas=n_columnas+1
    end
    
    # Crear un nuevo array bidimensional con ceros
    nuevo_array = zeros(T, n_filas, n_columnas)
    
    # Copiar los valores del array original al nuevo array, desplazándolos
    for i in 1:veces
        nuevo_array[i,1]=array[1,i]
    end
    for i in 1:n_filas
            for j in 2:n_columnas-1
                j_aux=(j-1)*(veces*3/4)
                j_aux=round(Int,j_aux)
                nuevo_array[i,j] = array[1,i+j_aux]
            end
    end
    for i in 1:veces
        nuevo_array[i,n_columnas]=array[1,(length(array))-veces+i]
    end
    return nuevo_array
end

function crear_wav(data::Vector{Float64}, fs::Int, filename::String)
    wavwrite(data, fs, filename)
end

#Crea una matriz que para cada muestra guarda mediate un 0 o un 1 que nota esta sonando que cada valor de cada muestra
function gen_target(filename::String, num_filas::Int, veces::Int)
    # Leer el archivo CSV
    df = CSV.File(filename) |> DataFrame
    valores=127#-recortar_valores
    matriz = zeros(Int, valores, num_filas)
    num_filas=num_filas#-(muestras_input)/2
    num_filas=round(Int,num_filas)
    #salto=round(Int,muestras_input*2/3)
    # Iterar sobre cada fila del DataFrame
    for fila in 1:size(df, 1)
        # Obtener los valores de start_time, end_time y note
        inicio = df[fila, :start_time]
        fin = df[fila, :end_time]
        columna = df[fila, :note]
        columna=columna#-recortar_valores
        if columna>0
            # Iterar sobre las filas y poner 1 en las columnas correspondientes
            for num_fila in inicio:fin
                if 1 <= num_fila <= num_filas
                    fila_real=(num_fila-1-(num_fila-1)%veces)/veces+1
                    fila_real=round(Int,fila_real)
                    matriz[ columna,fila_real] = 1
                    fila_real=(num_fila-(num_fila+veces*3/4)%veces+veces*3/4)/veces+1
                    fila_real=round(Int,fila_real)
                    #if fila_real>0
                    matriz[ columna,fila_real] = 1
                    #end
                end
            end 
        end
    end
    return matriz
end

function pruebas()
    # Ruta del archivo .wav
    archivo_wav = "/home/pajon/Escritorio/Programacion/3º_2c/AA/Practica/train_data/01.wav"

    # Obtener información del archivo WAV
    canales, frecuencia_muestreo, tasa_bits_codificacion = obtener_info_wav(archivo_wav)
    println(frecuencia_muestreo)

    #prueba conversion para crear ventanas
    l=[1 2 3 4 5 6 7 8]
    println(convertir_array(l,4))

    # Cargar el archivo .wav y crear el input 
    input = gen_input(archivo_wav)
    input2=convertir_array(input, muestras_input)
    # Ver la forma del input creado
    println("Forma del input: ", size(input2))
    #crear_wav(input[1,:],8000,"prueba.wav")
    println(size(gen_target("/home/pajon/Escritorio/Programacion/3º_2c/AA/Practica/train_labels/01.csv",round(Int,frecuencia_muestreo),8000)))
end 

function procesar_archivos(input_folder::String, target_folder::String)
    # Obtener la lista de archivos en el directorio de entrada
    input_files = readdir(input_folder)
    target_files = readdir(target_folder)

    # Filtrar solo los archivos de audio WAV
    input_files = filter(x -> endswith(x, ".wav"), input_files)

    for input_file in input_files
        # Verificar si hay un archivo CSV correspondiente
        filename_base = splitext(input_file)[1]  # Obtener el nombre base sin la extensión
        matching_csv = joinpath(target_folder, filename_base * ".csv")
        #println(matching_csv)
        if isdir(input_folder) && isdir(target_folder)
            # Si hay un archivo CSV correspondiente, cargar ambos archivos
            input_wav = joinpath(input_folder, input_file)
            target_csv = joinpath(target_folder, filename_base * ".csv")

            # Aquí puedes realizar tu tarea de procesamiento de datos
            entrenar(input_wav,target_csv)

        else
            println("No se encontró un archivo CSV correspondiente para $input_file")
        end
    end
end

function entrenar(input_wav::String,target_csv::String)
    canales, frecuencia_muestreo, tasa_bits_codificacion,tamano = obtener_info_wav(input_wav)
    input = gen_input(input_wav)
    input=convertir_array(input, muestras_input)
    #println(input_wav)
    #println(size(input))
    #for i in 1:size(input,2)
    #    println(size(input[:,i]))
    #    crear_wav(input[:,i],44100,"/home/pajon/Escritorio/Programacion/3º_2c/AA/Practica/example/"*string(i)*".wav")
    #end
    target=gen_target(target_csv,size(input,2),muestras_input)
    println(target)
    println()
    #red 
end

entrenar("/home/pajon/Escritorio/Programacion/3º_2c/AA/Practica/train_data/01.wav","/home/pajon/Escritorio/Programacion/3º_2c/AA/Practica/train_labels/01.csv")    
#procesar_archivos("/home/pajon/Escritorio/Programacion/3º_2c/AA/Practica/train_data","/home/pajon/Escritorio/Programacion/3º_2c/AA/Practica/train_labels")