class Config:
	#Preprocessing
    MAX_ARTICLES = 10000
    INPUT_FILE_DATA_TRAIN = '../data/articles-training-20180831.xml/articles-training-20180831.xml'
    INPUT_FILE_DATA_TRAIN_GROUDTRUTH = '../data/ground-truth-training-20180831.xml/ground-truth-training-20180831.xml'
    FILE_FREQ_ID = 'clean_articles.csv'
    FILE_TRAIN = 'train.csv'
    TOP_WORDS = 5000

    #Model
    EMBEDDING_VECTOR_LEN = 32
    MAX_REVIEW_LEN = 500