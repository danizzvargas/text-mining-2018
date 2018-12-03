# Regresion Logistica

## Informacion
Uso de regresion logistica para clasificacion de textos hiper-partidistas. 
Se usan las caracteristicas de:
1. Articulo
2. Numero de parrafos
3. Numero de links

## Instrucciones
1. Verificar las configuraciones de los archivos de entrenamiento, validacion y test en config.py
2. Correr idToLabel.py para la obtencion de un mapa de articulo->clasificacion usando los datos groudTruth
⋅⋅1. python idToLabel.py train
⋅⋅2. python idToLabel.py validate
⋅⋅3. python idToLabel.py test
3. Correr preprocessing.py para pre-procesar los datos
4. Correr logRegresion.py

## Resultados
Entrenamiento 200,000 datos

En validacion de datos por publisher:

| Accuracy | Precision | Recall  | F1  |
|----------|-----------|---------|-----|
|0.572525|0.54804|0.826523|0.65907|

En validacion de datos manuales:

| Accuracy | Precision | Recall  | F1  |
|----------|-----------|---------|-----|
|0.528682|0.4357976|0.9411|0.595744|
