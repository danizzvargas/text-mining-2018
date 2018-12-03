class Config:
	#Preprocessing
    MAX_ARTICLES = 10000*20
    MAX_ARTICLES_VAL = 2000*20
    INPUT_FILE_DATA_TRAIN = '../data/articles-training-bypublisher-20181122/articles-training-bypublisher-20181122.xml'
    INPUT_FILE_DATA_TRAIN_GROUDTRUTH = '../data/ground-truth-training-bypublisher-20181122/ground-truth-training-bypublisher-20181122.xml'
    INPUT_FILE_DATA_VAL = '../data/articles-validation-bypublisher-20181122/articles-validation-bypublisher-20181122.xml'
    INPUT_FILE_DATA_VAL_GROUDTRUTH = '../data/ground-truth-validation-bypublisher-20181122/ground-truth-validation-bypublisher-20181122.xml'
    INPUT_FILE_DATA_TEST = '../data/articles-training-byarticle-20181122/articles-training-byarticle-20181122.xml'
    INPUT_FILE_DATA_TEST_GROUDTRUTH = '../data/ground-truth-training-byarticle-20181122/ground-truth-training-byarticle-20181122.xml'
    FILE_FREQ_ID = 'clean_articles.csv'
    FILE_FREQ_ID_VAL = 'clean_articles_val.csv'
    FILE_FREQ_ID_TEST = 'clean_articles_test.csv'
    FILE_TRAIN = 'train.csv'
    FILE_VAL = 'validate.csv'
    FILE_TEST = 'test.csv'

    #Model
