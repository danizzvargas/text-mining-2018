# text-mining-2018

## Instrucciones

El repositorio tiene varios modelos para la clasificacion de textos hiperpartidistas. 
Para cada modelo se tienen distintas instrucciones para su preprocesamiento, entrenamietno e inferencia.

Todos los modelos comparten el mismo dataset, por lo que es necesario descargar los datos del [Semval]
(https://pan.webis.de/semeval19/semeval19-web/) a distintos folders, por ejemplo:
```
INPUT_FILE_DATA_TRAIN = 'articles-training-20180831.xml/articles-training-20180831.xml'
INPUT_FILE_DATA_TRAIN_GROUDTRUTH = 'data/ground-truth-training-20180831.xml/ground-truth-training-20180831.xml'
INPUT_FILE_DATA_VAL = 'data/articles-validation-20180831.xml/articles-validation-20180831.xml'
INPUT_FILE_DATA_VAL_GROUDTRUTH = 'data/ground-truth-validation-20180831.xml/ground-truth-validation-20180831.xml'
```
## Modelos
1. Regresion Logistica : [Instrucciones](LogReg.md)
2. GRU con Glove Embeddings : [Instrucciones](GruGlove2.md)
3. GRU con Word2Vec 
4. CNN : [Instrucciones](CNN.md)
