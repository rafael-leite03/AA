using WAV
using Flux
using SampledSignals
using CSV
using DataFrames
using FFTW
using Statistics
using Plots
using PlotlyJS
using Flux.Losses

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

function convertir_a_array(array::Vector{T}, veces::Int) where T
    # Calcular el tamaño del nuevo array
    n_filas = 1
    n_columnas = ceil(Int, length(array) / veces)
    
    # Crear un nuevo array bidimensional con ceros
    nuevo_array = zeros(T, n_filas, n_columnas)
    
    # Copiar los valores del array original al nuevo array
    for i in 1:n_columnas
        inicio = (i - 1) * veces + 1
        fin = min(i * veces, length(array))
        nuevo_array[1, i] = mean(@view(array[inicio:fin]))
    end
    
    return nuevo_array
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
            #entrenar(input_wav,target_csv)

        else
            println("No se encontró un archivo CSV correspondiente para $input_file")
        end
    end
end

function entrenar(input_wav1::String,input_wav2::String,f1ini::Int,f1fin::Int,f2ini::Int,f2fin::Int)
    canales, frecuencia_muestreo, tasa_bits_codificacion,tamano = obtener_info_wav(input_wav1)
    input = gen_input(input_wav1)
    input1=convertir_array(input, muestras_input)
    input = gen_input(input_wav2)
    input2=convertir_array(input, muestras_input)
    input=zeros(size(input1,2)+size(input2,2),4)
    target=zeros(Int,size(input1,2)+size(input2,2),2)
    for i in 1:size(input1,2)
        aux=reshape(input1[:,i], 1, length(input1[:,i]))
        #println(size(aux))
        input[i,1],input[i,2]=mirar2notas(aux,frecuencia_muestreo,f1ini,f1fin);
        input[i,3],input[i,4]=mirar2notas(aux,frecuencia_muestreo,f2ini,f2fin);
        target[i,1]=1;
    end
    canales, frecuencia_muestreo, tasa_bits_codificacion,tamano = obtener_info_wav(input_wav2)
    for i in 1:size(input2,2)
        aux=reshape(input2[:,i], 1, length(input2[:,i]))
        #println(size(aux))
        input[i+size(input1,2),1],input[i+size(input1,2),2]=mirar2notas(aux,frecuencia_muestreo,f1ini,f1fin);
        input[i+size(input1,2),3],input[i+size(input1,2),4]=mirar2notas(aux,frecuencia_muestreo,f2ini,f2fin);
        target[i+size(input1,2),2]=1;
    end
    
    ann = Chain(
    Dense(4, 5, σ),
    Dense(5, 3, σ),
    Dense(3, 2, identity),softmax );
    loss(model, x, y) = Losses.crossentropy(model(x), y)
    learningRate=0.1
    opt_state = Flux.setup(Adam(learningRate), ann)
    Flux.train!(loss, ann, [(input', target')], opt_state)
    println(ann(input[1,:]))
    println(loss)
    
end

function mirar2notas(input::Matrix{Float64},frecuencia::Float32,note1::Int, note2::Int)

    #canales, Fs, tasa_bits_codificacion,tamano = obtener_info_wav(input_wav)
    #input_org = gen_input(input_wav)
    #input=convertir_array(input_org, muestras_input)
    #println(size(input_org,2))
    # Numero de muestras
    
    n = size(input,2);
    # Que frecuenicas queremos coger
    f1 = note1; f2 = note2;
    Fs=frecuencia;

    #println("$(n) muestras con una frecuencia de $(Fs) muestras/seg: $(n/Fs) seg.")

    # Creamos una señal de n muestras: es un array de flotantes
    x = 1:n;
    senalTiempo = input[1,:];
    
    
    # Representamos la señal
    #plotlyjs();
    #graficaTiempo = plot(x, senalTiempo, label = "", xaxis = x);
    
    # Hallamos la FFT y tomamos el valor absoluto
    senalFrecuencia = abs.(fft(senalTiempo));
    
    
    
    # Los valores absolutos de la primera mitad de la señal deberian de ser iguales a los de la segunda mitad, salvo errores de redondeo
    # Esto se puede ver en la grafica:
    #graficaFrecuencia = plot(senalFrecuencia, label = "");
    #  pero ademas lo comprobamos en el codigo
    if (iseven(n))
        @assert(mean(abs.(senalFrecuencia[2:Int(n/2)] .- senalFrecuencia[end:-1:(Int(n/2)+2)]))<1e-8);
        senalFrecuencia = senalFrecuencia[1:(Int(n/2)+1)];
    else
        @assert(mean(abs.(senalFrecuencia[2:Int((n+1)/2)] .- senalFrecuencia[end:-1:(Int((n-1)/2)+2)]))<1e-8);
        senalFrecuencia = senalFrecuencia[1:(Int((n+1)/2))];
    end;
    
    # Grafica con la primera mitad de la frecuencia:
    #graficaFrecuenciaMitad = plot(senalFrecuencia, label = "");
    
    
    # Representamos las 3 graficas juntas
    #display(plot(graficaTiempo, graficaFrecuencia, graficaFrecuenciaMitad, layout = (3,1)));
    
    
    # A que muestras se corresponden las frecuencias indicadas
    #  Como limite se puede tomar la mitad de la frecuencia de muestreo
    m1 = Int(round(f1*2*length(senalFrecuencia)/Fs));
    m2 = Int(round(f2*2*length(senalFrecuencia)/Fs));
    
    # Unas caracteristicas en esa banda de frecuencias
    #println("Media de la señal en frecuencia entre $(f1) y $(f2) Hz: ", mean(senalFrecuencia[m1:m2]));
    #println("Desv tipica de la señal en frecuencia entre $(f1) y $(f2) Hz: ", std(senalFrecuencia[m1:m2]));
    return mean(senalFrecuencia[m1:m2]),std(senalFrecuencia[m1:m2])
end



canales, Fs, tasa_bits_codificacion,tamano = obtener_info_wav("/home/pajon/Escritorio/Programacion/3º_2c/AA/practica/train_data/01.wav")
#println(size(gen_input("/home/pajon/Escritorio/Programacion/3º_2c/AA/practica/train_data/01.wav")))
#mirar2notas(gen_input("/home/pajon/Escritorio/Programacion/3º_2c/AA/practica/train_data/01.wav"),Fs,10000,20000)
entrenar("/home/pajon/Escritorio/Programacion/3º_2c/AA/practica/train_data/01.wav","/home/pajon/Escritorio/Programacion/3º_2c/AA/practica/train_data/55.wav",10,20,80,90)    
#procesar_archivos("/home/pajon/Escritorio/Programacion/3º_2c/AA/practica/train_data","/home/pajon/Escritorio/Programacion/3º_2c/AA/practica/train_labels")