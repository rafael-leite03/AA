using WAV
using Flux
using SampledSignals
using CSV
using DataFrames
using FFTW
using Statistics
using Plots
using Random
using Flux.Losses
using ScikitLearn
using DelimitedFiles
using StatsPlots
using Statistics
@sk_import svm:SVC
@sk_import tree:DecisionTreeClassifier
@sk_import neighbors:KNeighborsClassifier


muestras_input=65536#potencia de 2 mayor o igual que cuatro
output_length=82
input_length=164
kfold=10

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
    tamano=size(array,2)
    # Copiar los valores del array original al nuevo array, desplazándolos
    for i in 1:veces
        if(i<=tamano)
            nuevo_array[i,1]=array[1,i]
        end
    end
    for i in 1:n_filas
            for j in 2:n_columnas-1
                j_aux=(j-1)*(veces*3/4)
                j_aux=round(Int,j_aux)
                nuevo_array[i,j] = array[1,i+j_aux]
            end
    end
    if(veces>tamano)
        veces=tamano
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

function procesar_archivos(input_folder::String)
    # Obtener la lista de carpetas en el directorio de entrada
    subfolders = readdir(input_folder)
    
    # Inicializar input y target
    input = nothing
    target = nothing
    value = 1
    #@assert(size(trainingInputs, 1) == size(trainingTargets, 1))
    #@assert(size(validationInputs, 1) == size(validationTargets, 1))
    #@assert(size(testInputs, 1) == size(testTargets, 1))
    folder_names = Float16[]
    for subfolder in subfolders
        push!(folder_names, parse(Float16, subfolder))
    end

    for subfolder in subfolders
        println(subfolder)
        subfolder_path = joinpath(input_folder, subfolder)
        
        # Verificar si el elemento es una carpeta
        if isdir(subfolder_path)
            # Obtener la lista de archivos en la subcarpeta actual
            input_files = readdir(subfolder_path)
            input_files = filter(x -> endswith(x, ".wav"), input_files)
            
            for input_file in input_files
                # Verificar si hay un archivo WAV correspondiente
                filename_base = splitext(input_file)[1]  # Obtener el nombre base sin la extensión
                matching_wav = joinpath(subfolder_path, filename_base * ".wav")
                
                if isfile(matching_wav)
                    # Si hay un archivo WAV correspondiente, cargar ambos archivos
                    input_wav = matching_wav
                    target_wav = matching_wav  # No hay target en el código proporcionado

                    # Aquí puedes realizar tu tarea de procesamiento de datos
                    if input === nothing && target === nothing
                        input_aux, target_aux = FFT_data(input_wav, value,folder_names)
                        input = input_aux
                        target = target_aux
                    else
                        input_aux, target_aux = FFT_data(input_wav, value,folder_names)
                        input = hcat(input, input_aux)
                        target = hcat(target, target_aux)
                    end

                    #println(target)                    

                else
                    println("No se encontró un archivo WAV correspondiente para $input_file")
                end
            end

        end
        value += 1
    end
    println(size(input))
    open("input.txt", "w") do archivo
        writedlm(archivo, input)
    end
    open("target.txt", "w") do archivo
        writedlm(archivo, target)
    end
end

function entrenar_RRNNAA(input_train,target_train,input_test,target_test)
    gr();
    minLoss=0.1
    learningRate=0.02
    input_validation=nothing
    target_validation=nothing
    max_ciclos=120
    iteracion=1
    restar=0
    historico_train=Float64[]
    historico_validation=Float64[]
    historico_test=Float64[]
    println(size(input_train))
    #println(size(input_train,2))
    #extrae del conjunto de entrenamiento el conjunto de test y validacion
    for i in 1:size(input_train,2)
        x=rand(1:10)
        if x%10==2||x%10==6
            if input_validation==nothing
                input_validation=zeros(input_length,1)
                input_train,input_validation[:,1]=extract_and_remove(input_train,i-restar)
            else
                aux=zeros(input_length,1)
                input_train,aux[:,1]=extract_and_remove(input_train,i-restar)
                input_validation=hcat(input_validation,aux)
            end
            if target_validation==nothing
                target_validation=zeros(output_length,1)
                target_train,target_validation[:,1]=extract_and_remove(target_train,i-restar)
            else
                aux=zeros(output_length,1)
                target_train,aux[:,1]=extract_and_remove(target_train,i-restar)
                target_validation=hcat(target_validation,aux)
            end
            restar+=1
        end
    end
    restar=0
    for i in 1:size(input_test,2)
        x=rand(1:10)
        if x%10==2||x%10==6
            if input_validation==nothing
                input_validation=zeros(input_length,1)
                input_test,input_validation[:,1]=extract_and_remove(input_test,i-restar)
            else
                aux=zeros(input_length,1)
                input_test,aux[:,1]=extract_and_remove(input_test,i-restar)
                input_validation=hcat(input_validation,aux)
            end
            if target_validation==nothing
                target_validation=zeros(output_length,1)
                target_test,target_validation[:,1]=extract_and_remove(target_test,i-restar)
            else
                aux=zeros(output_length,1)
                target_test,aux[:,1]=extract_and_remove(target_test,i-restar)
                target_validation=hcat(target_validation,aux)
            end
            restar+=1
        end
    end
    ann = Chain(
        Dense(input_length, 130, σ),
        Dense(130, 110, σ),
        Dense(110, 90, σ),
        Dense(90, output_length, identity),softmax );
    
    loss(model,x, y) = Losses.crossentropy(model(x), y)    
    opt_state = Flux.setup(Adam(learningRate), ann)    
    outputP = ann(input_train)
    vlose = loss(ann,input_validation, target_validation)
    mejor=vlose
    #while (vlose > minLoss&&max_ciclos>iteracion&&sin_mejora!=parar_nomejora)
    while (vlose > minLoss&&max_ciclos>iteracion)

        Flux.train!(loss, ann, [(input_train, target_train)], opt_state)  
        vlose = loss(ann,input_validation, target_validation)
        outputP = ann(input_validation)
        #vacc = accuracy(outputP, target_validation)
        #if(vlose>mejor)
        #    sin_mejora+=1
        #else
        #    mejor=vlose
        #    sin_mejora=0
        #end
        push!(historico_train,loss(ann,input_train, target_train))
        push!(historico_test,loss(ann,input_test, target_test))
        push!(historico_validation,vlose)
        #println(vlose)
        iteracion+=1
    end
    
    vlose = loss(ann,input_test, target_test)
    outputP = ann(input_test)
    #vacc = accuracy(outputP, target_test)
    push!(historico_train,loss(ann,input_train, target_train))
    push!(historico_test,loss(ann,input_test, target_test))
    push!(historico_validation,vlose)
    p1=plot(historico_train, title="Historico Train", subplot=1)
    p2=plot(historico_test, title="Historico Test", subplot=1)
    p3=plot(historico_validation, title="Historico Validation", subplot=1)
    display(plot(p1,p2,p3, layout = (3,1)));   
    # Convertir los valores en bool dependiendo si son mayores o menores que 0.5
    outputP = nearest_to_one_matrix(outputP)
    outputP = Array{Bool}(outputP .> 0.5)
    # Convertir los valores en bool dependiendo si son mayores o menores que 0.5
    target_test = nearest_to_one_matrix(target_test)
    target_test = Array{Bool}(target_test .> 0.5)
    #printConfusionMatrix(outputP',target_test')
    return outputP',target_test',vlose
    
end

function nearest_to_one_matrix(matrix::Matrix)
    result = similar(matrix, Bool)

    for j in 1:size(matrix, 2)
        min_distance = Inf
        nearest_index = -1

        # Encontrar el índice del valor más cercano a 1 en la columna j
        for i in 1:size(matrix, 1)
            distance = abs(1 - matrix[i, j])
            if distance < min_distance
                min_distance = distance
                nearest_index = i
            end
        end

        # Configurar a 1 el valor más cercano a 1 y el resto a 0 en la columna j
        result[nearest_index, j] = 1
        for i in 1:size(matrix, 1)
            if i != nearest_index
                result[i, j] = 0
            end
        end
    end

    return result
end

function entrenar_svm(input_train, target_train,input_test, target_test)
    println(size(input_train))
    model = SVC(kernel="rbf", degree=5, gamma=3, C=2);
    fit!(model, input_train', crear_vector(target_train)');
    testOutputs = predict(model, input_test');
    #aux=crear_vector(target_test)'
    #println(aux)
    #printConfusionMatrix(testOutputs,collect(aux))
    aux=(Array{Bool}(target_test .== 1))'
    #printConfusionMatrix(recrear_vector(testOutputs,output_length),aux)
    mae = mean(abs.(testOutputs .- target_test'))
    return recrear_vector(testOutputs,output_length),aux,mae
end

function entrenar_tree(input_train, target_train,input_test, target_test)
    println(size(input_train))
    model = DecisionTreeClassifier(max_depth=2, random_state=1)
    fit!(model, input_train', crear_vector(target_train)');
    testOutputs = predict(model, input_test');
    aux=(Array{Bool}(target_test .> 0.5))'
    #printConfusionMatrix(recrear_vector(testOutputs,output_length),aux)
    # Calcular el error absoluto medio (MAE)
    mae = mean(abs.(testOutputs .- target_test'))
    return recrear_vector(testOutputs,output_length),aux,mae
end

function entrenar_KNe(input_train, target_train,input_test, target_test)
    println(size(input_train))
    model = KNeighborsClassifier(2);
    fit!(model, input_train', crear_vector(target_train)');
    testOutputs = predict(model, input_test');
    aux=(Array{Bool}(target_test .> 0.5))'
    #printConfusionMatrix(recrear_vector(testOutputs,output_length),aux)
    mae = mean(abs.(testOutputs .- target_test'))
    return recrear_vector(testOutputs,output_length),aux,mae
end

function crear_vector(matriz::Matrix{Float64})
    m, n = size(matriz)
    vector_resultado = zeros(Int, n)
    
    for i in 1:m
        for j in 1:n
            if matriz[i, j] == 1
                vector_resultado[j] = i
            end
        end
    end
    
    return vector_resultado
end

function recrear_vector(array,size::Int)
    #println(array)
    result=zeros(length(array),size)
    for i in 1:length(array)
        result[i,array[i]]=1
    end
    #println(result)
    return Array{Bool}(result .> 0.5)
    #return Array{Bool}(array .> 0.5)
end

function FFT_data(input_wav1::String, value::Int, folder_names)
    # El target de debe cambiar
    canales, frecuencia_muestreo, tasa_bits_codificacion, tamano = obtener_info_wav(input_wav1)
    input = gen_input(input_wav1)
    input1 = convertir_array(input, muestras_input)
        
    # Preinicializar las matrices con el tamaño adecuado
    input_size = size(input1, input_length)
    inputs = Matrix{Float64}(undef, input_length, input_size)
    targets = Matrix{Float64}(undef, output_length, input_size)

    for i in 1:input_size
        aux = reshape(input1[:, i], 1, length(input1[:, i]))
        input_aux = zeros(input_length, 1)
        input_aux[:, 1] .= mirartodasnotas(aux, frecuencia_muestreo, folder_names)
        target_aux = zeros(output_length, 1)
        target_aux[value, 1] = 1.0
        inputs[:, i] .= input_aux
        targets[:, i] .= target_aux
    end

    return inputs, targets
end

function mirartodasnotas(input::Matrix{Float64},frecuencia::Float32,folder_names)
    res = zeros(Float64, 2 * length(folder_names)) 
    i=1
    for note in folder_names
        res[i],res[i+1]=mirar2notas(input,frecuencia,note)
        i=i+2
    end
    return res
end

function extract_and_remove(matrix::AbstractMatrix, col_index::Integer)
    if 1 <= col_index <= size(matrix, 2)
        col = matrix[:, col_index]
        matrix = hcat(matrix[:, 1:col_index-1], matrix[:, col_index+1:end])
        #println(size(matrix))
        return matrix, col
    else
        throw(ArgumentError("Índice de columna fuera de rango"))
    end
end



function mirar2notas(input::Matrix{Float64},frecuencia::Float32,note::Float16)

    #canales, Fs, tasa_bits_codificacion,tamano = obtener_info_wav(input_wav)
    #input_org = gen_input(input_wav)
    #input=convertir_array(input_org, muestras_input)
    #println(size(input_org,2))
    # Numero de muestras
    
    n = size(input,2);
    # Que frecuenicas queremos coger
    f1 = note*0.95; f2=note*1.05;
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

    # recortamos la mitad no necesaria

    m1 = Int(round(f1*2*length(senalFrecuencia)/Fs));
    m2 = Int(round(f2*2*length(senalFrecuencia)/Fs));
    
    # Unas caracteristicas en esa banda de frecuencias
    #println("Media de la señal en frecuencia entre $(f1) y $(f2) Hz: ", mean(senalFrecuencia[m1:m2]));
    #println("Desv tipica de la señal en frecuencia entre $(f1) y $(f2) Hz: ", std(senalFrecuencia[m1:m2]));
    return mean(senalFrecuencia[m1:m2]),std(senalFrecuencia[m1:m2])
end

accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1}) = mean(outputs.==targets);
function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2})
    @assert(all(size(outputs).==size(targets)));
    if (size(targets,2)==1)
        return accuracy(outputs[:,1], targets[:,1]);
    else
        return mean(all(targets .== outputs, dims=2));
    end;
end;

function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    @assert(size(outputs)==size(targets));
    numClasses = size(targets,2);
    # Nos aseguramos de que no hay dos columnas
    @assert(numClasses!=2);
    if (numClasses==1)
        return confusionMatrix(outputs[:,1], targets[:,1]);
    end;

    # Nos aseguramos de que en cada fila haya uno y sólo un valor a true
    @assert(all(sum(outputs, dims=2).==1));
    # Reservamos memoria para las metricas de cada clase, inicializandolas a 0 porque algunas posiblemente no se calculen
    recall      = zeros(numClasses);
    specificity = zeros(numClasses);
    precision   = zeros(numClasses);
    NPV         = zeros(numClasses);
    F1          = zeros(numClasses);
    # Calculamos el numero de patrones de cada clase
    numInstancesFromEachClass = vec(sum(targets, dims=1));
    # Calculamos las metricas para cada clase, esto se haria con un bucle similar a "for numClass in 1:numClasses" que itere por todas las clases
    #  Sin embargo, solo hacemos este calculo para las clases que tengan algun patron
    #  Puede ocurrir que alguna clase no tenga patrones como consecuencia de haber dividido de forma aleatoria el conjunto de patrones entrenamiento/test
    #  En aquellas clases en las que no haya patrones, los valores de las metricas seran 0 (los vectores ya estan asignados), y no se tendran en cuenta a la hora de unir estas metricas
    for numClass in findall(numInstancesFromEachClass.>0)
        # Calculamos las metricas de cada problema binario correspondiente a cada clase y las almacenamos en los vectores correspondientes
        (_, _, recall[numClass], specificity[numClass], precision[numClass], NPV[numClass], F1[numClass], _) = confusionMatrix(outputs[:,numClass], targets[:,numClass]);
    end;

    # Reservamos memoria para la matriz de confusion
    confMatrix = Array{Int64,2}(undef, numClasses, numClasses);
    # Calculamos la matriz de confusión haciendo un bucle doble que itere sobre las clases
    for numClassTarget in 1:numClasses, numClassOutput in 1:numClasses
        # Igual que antes, ponemos en las filas los que pertenecen a cada clase (targets) y en las columnas los clasificados (outputs)
        confMatrix[numClassTarget, numClassOutput] = sum(targets[:,numClassTarget] .& outputs[:,numClassOutput]);
    end;

    # Aplicamos las forma de combinar las metricas macro o weighted
    if weighted
        # Calculamos los valores de ponderacion para hacer el promedio
        weights = numInstancesFromEachClass./sum(numInstancesFromEachClass);
        recall      = sum(weights.*recall);
        specificity = sum(weights.*specificity);
        precision   = sum(weights.*precision);
        NPV         = sum(weights.*NPV);
        F1          = sum(weights.*F1);
    else
        # No realizo la media tal cual con la funcion mean, porque puede haber clases sin instancias
        #  En su lugar, realizo la media solamente de las clases que tengan instancias
        numClassesWithInstances = sum(numInstancesFromEachClass.>0);
        recall      = sum(recall)/numClassesWithInstances;
        specificity = sum(specificity)/numClassesWithInstances;
        precision   = sum(precision)/numClassesWithInstances;
        NPV         = sum(NPV)/numClassesWithInstances;
        F1          = sum(F1)/numClassesWithInstances;
    end;
    # Precision y tasa de error las calculamos con las funciones definidas previamente
    acc = accuracy(outputs, targets);
    errorRate = 1 - acc;

    return (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix);
end;

function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    numInstances = length(targets);
    @assert(length(outputs)==numInstances);
    # Valores de la matriz de confusion
    TN = sum(.!outputs .& .!targets); # Verdaderos negativos
    FN = sum(.!outputs .&   targets); # Falsos negativos
    TP = sum(  outputs .&   targets); # Verdaderos positivos
    FP = sum(  outputs .& .!targets); # Falsos negativos
    # Creamos la matriz de confusión, poniendo en las filas los que pertenecen a cada clase (targets) y en las columnas los clasificados (outputs)
    #  Primera fila/columna: negativos
    #  Segunda fila/columna: positivos
    confMatrix = [TN FP; FN TP];
    # Metricas que se derivan de la matriz de confusion:
    acc         = (TN+TP)/(TN+FN+TP+FP);
    errorRate   = 1. - acc;
    # Para sensibilidad, especificidad, VPP y VPN controlamos que algunos casos pueden ser NaN
    #  Para el caso de sensibilidad y especificidad, en un conjunto de entrenamiento estos no pueden ser NaN, porque esto indicaria que se ha intentado entrenar con una unica clase
    #   Sin embargo, sí pueden ser NaN en el caso de aplicar un modelo en un conjunto de test, si este sólo tiene patrones de una clase
    #  Para VPP y VPN, sí pueden ser NaN en caso de que el clasificador lo haya clasificado todo como negativo o positivo respectivamente
    # En estos casos, estas metricas habria que dejarlas a NaN para indicar que no se han podido evaluar
    #  Sin embargo, como es posible que se quiera combinar estos valores al evaluar una clasificacion multiclase, es necesario asignarles un valor. El criterio que se usa aqui es que estos valores seran igual a 0
    # Ademas, hay un caso especial: cuando los VP son el 100% de los patrones, o los VN son el 100% de los patrones
    #  En este caso, el sistema ha actuado correctamente, así que controlamos primero este caso
    if (TN==numInstances) || (TP==numInstances)
        recall = 1.;
        precision = 1.;
        specificity = 1.;
        NPV = 1.;
    else
        recall      = (TP==TP==0.) ? 0. : TP/(TP+FN); # Sensibilidad
        specificity = (TN==FP==0.) ? 0. : TN/(TN+FP); # Especificidad
        precision   = (TP==FP==0.) ? 0. : TP/(TP+FP); # Valor predictivo positivo
        NPV         = (TN==FN==0.) ? 0. : TN/(TN+FN); # Valor predictivo negativo
    end;
    # Calculamos F1, teniendo en cuenta que si sensibilidad o VPP es NaN (pero no ambos), el resultado tiene que ser 0 porque si sensibilidad=NaN entonces VPP=0 y viceversa
    F1          = (recall==precision==0.) ? 0. : 2*(recall*precision)/(recall+precision);
    return (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix)
end;



function printConfusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix) = confusionMatrix(outputs, targets; weighted=weighted);
    numClasses = size(confMatrix,1);
    writeHorizontalLine() = (for i in 1:numClasses+1 print("--------") end; println(""); );
    writeHorizontalLine();
    print("\t| ");
    if (numClasses==2)
        println(" - \t + \t|");
    else
        print.("Cl. ", 1:numClasses, "\t| ");
    end;
    println("");
    writeHorizontalLine();
    for numClassTarget in 1:numClasses
        # print.(confMatrix[numClassTarget,:], "\t");
        if (numClasses==2)
            print(numClassTarget == 1 ? " - \t| " : " + \t| ");
        else
            print("Cl. ", numClassTarget, "\t| ");
        end;
        print.(confMatrix[numClassTarget,:], "\t| ");
        println("");
        writeHorizontalLine();
    end;
    println("Accuracy: ", acc);
    println("Error rate: ", errorRate);
    println("Recall: ", recall);
    println("Specificity: ", specificity);
    println("Precision: ", precision);
    println("Negative predictive value: ", NPV);
    println("F1-score: ", F1);
    return (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix);
end;


function printConfusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix) = confusionMatrix(outputs, targets)

    println("Confusion Matrix:")
    println(confMatrix)
    println("Accuracy: $acc")
    println("Error Rate: $errorRate")
    println("Recall (Sensitivity): $recall")
    println("Specificity: $specificity")
    println("Precision (Positive Predictive Value): $precision")
    println("Negative Predictive Value: $NPV")
    println("F1 Score: $F1")
    println(" ")

end

function calcular_accuracy_por_clase(resultados, targets)
    num_muestras, num_clases = size(targets)
    accuracy_por_clase = similar(1:num_clases, Float64) # Vector para almacenar la precisión por clase
    
    for clase in 1:num_clases
        # Obtener los resultados y objetivos para esa clase
        targets_clase = targets[:, clase]
        resultados_clase = resultados[findall(targets_clase), clase]  # Solo considerar las muestras donde la clase es verdadera
        
        # Calcular la precisión para la clase específica
        accuracy_por_clase[clase] = sum(resultados_clase .== targets_clase[findall(targets_clase)]) / length(targets_clase[findall(targets_clase)])
    end

    return accuracy_por_clase
end


function ejecutar_crosscalidation(input,target,arquitectura)
    columnas_totales = size(input, 2)
    indices = collect(1:columnas_totales)
    output_data=nothing
    target_data=nothing
    error_data=[]
    lose=0
    veces=0
    Random.shuffle!(indices)
    kfold_size=round(Int,columnas_totales/kfold+1)

    for i in 1:kfold_size:columnas_totales
        grupo_actual = min(i + (kfold_size-1), columnas_totales)  # Asegura que el último grupo no exceda el tamaño total
        columnas_grupo = indices[i:grupo_actual]
        columnas_restantes = setdiff(indices, columnas_grupo)
        if output_data==nothing
            if arquitectura == 1
                output_data, target_data, lose_aux = entrenar_RRNNAA(input[:, columnas_restantes], target[:, columnas_restantes], input[:, columnas_grupo], target[:, columnas_grupo])
            elseif arquitectura == 2
                output_data, target_data, lose_aux = entrenar_tree(input[:, columnas_restantes], target[:, columnas_restantes], input[:, columnas_grupo], target[:, columnas_grupo])
            elseif arquitectura == 3
                output_data, target_data, lose_aux = entrenar_svm(input[:, columnas_restantes], target[:, columnas_restantes], input[:, columnas_grupo], target[:, columnas_grupo])
            elseif arquitectura == 4
                output_data, target_data, lose_aux = entrenar_KNe(input[:, columnas_restantes], target[:, columnas_restantes], input[:, columnas_grupo], target[:, columnas_grupo])
            else
                println("Estado no válido")
            end
            lose=lose_aux+lose
            push!(error_data, accuracy(output_data,target_data))
        else
            if arquitectura == 1
                output_aux, target_aux, lose_aux = entrenar_RRNNAA(input[:, columnas_restantes], target[:, columnas_restantes], input[:, columnas_grupo], target[:, columnas_grupo])
            elseif arquitectura == 2
                output_aux, target_aux, lose_aux = entrenar_tree(input[:, columnas_restantes], target[:, columnas_restantes], input[:, columnas_grupo], target[:, columnas_grupo])
            elseif arquitectura == 3
                output_aux, target_aux, lose_aux = entrenar_svm(input[:, columnas_restantes], target[:, columnas_restantes], input[:, columnas_grupo], target[:, columnas_grupo])
            elseif arquitectura == 4
                output_aux, target_aux, lose_aux = entrenar_KNe(input[:, columnas_restantes], target[:, columnas_restantes], input[:, columnas_grupo], target[:, columnas_grupo])
            else
                println("Estado no válido")
            end
            lose=lose+lose_aux
            push!(error_data, accuracy(output_aux,target_aux))
            output_data=vcat(output_data,output_aux)
            target_data=vcat(target_data,target_aux)
            
        end
        veces=veces+1
    end

    function matriz_bools_a_clases(matriz)
        clases = []
        for fila in eachrow(matriz)
            push!(clases, findfirst(fila))
        end
        return clases
    end

    # Convertir la matriz a un vector de clases
    clases = matriz_bools_a_clases(target_data)

    # Calcular la cantidad de datos por clase
    conteo_clases = Dict{Int, Int}()
    for c in clases
        conteo_clases[c] = get(conteo_clases, c, 0) + 1
    end


    # Ordenar el conteo por clases
    clases_ordenadas = sort(collect(keys(conteo_clases)))
    cantidad_datos = [get(conteo_clases, c, 0) for c in clases_ordenadas]

    # Crear la gráfica de barras
    p=bar(clases_ordenadas, cantidad_datos, xlabel="Clase", ylabel="Cantidad de datos", 
        title="Cantidad de datos por clase")
    display(p)

    #println(error_data)

    #error_data=calcular_mse_por_clase(output_data, target_data)
    println(size(output_data))
    error_data = replace(error_data, NaN => 0.0)
    #gr();
    #p = boxplot( error_data, xlabel="Class", ylabel="Mean Squared Error", title="Boxplot of Mean Squared Error per Class", size=(1920, 1080)) # Tamaño ajustado    
    #display(p)

    # Calcula la desviación típica
    suma_cuadrados = sum((output_data .- target_data).^2)
    N = length(output_data)
    desviacion_tipica = sqrt(suma_cuadrados / N)
    #println("-RRNNAA:")
    printConfusionMatrix(output_data,target_data)
    println("Desviacion tipica: ",desviacion_tipica)
    println("Error: ",(lose/veces))
    println()
    return error_data
    
end

error_data=nothing

#procesar_archivos("carpeta_input");

archivo = open("input.txt", "r")
input = readdlm(archivo)
close(archivo)

archivo = open("target.txt", "r")
target = readdlm(archivo)
close(archivo)

error_data=ejecutar_crosscalidation(input,target,1)
error_data=hcat(error_data,ejecutar_crosscalidation(input,target,2))
error_data=hcat(error_data,ejecutar_crosscalidation(input,target,3))
error_data=hcat(error_data,ejecutar_crosscalidation(input,target,4))
gr();
println(size(error_data))
error_data = replace(error_data, NaN => 0.0)
error_data = convert(Matrix{Float64}, error_data)
arquitectura_labels = ["arquitectura $i" for i in 1:4]
p = boxplot(error_data, xlabel="Class", ylabel="Mean Accuracy", title="Boxplot of Accuracy per model", size=(1920, 1080)) # Tamaño ajustado    
display(p)