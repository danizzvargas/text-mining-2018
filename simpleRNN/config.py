class Config:
	#Preprocessing
    MAX_ARTICLES = 100000
    MAX_ARTICLES_VAL = 20000
    INPUT_FILE_DATA_TRAIN = '../data/articles-training-20180831.xml/articles-training-20180831.xml'
    INPUT_FILE_DATA_TRAIN_GROUDTRUTH = '../data/ground-truth-training-20180831.xml/ground-truth-training-20180831.xml'
    INPUT_FILE_DATA_VAL = '../data/articles-validation-20180831.xml/articles-validation-20180831.xml'
    INPUT_FILE_DATA_VAL_GROUDTRUTH = '../data/ground-truth-validation-20180831.xml/ground-truth-validation-20180831.xml'
    FILE_FREQ_ID = 'clean_articles.csv'
    FILE_FREQ_ID_VAL = 'clean_articles_val.csv'
    FILE_TRAIN = 'train.csv'
    FILE_VAL = 'validate.csv'
    TOP_WORDS = 5000

    #Model
    EMBEDDING_VECTOR_LEN = 32
    MAX_REVIEW_LEN = 500