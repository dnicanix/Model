import pandas as pd
import xlsxwriter
import collections
import string
import utils.Normalizer as Normalizer
from sklearn.feature_extraction import stop_words
from porter2stemmer import Porter2Stemmer
from nltk.stem import PorterStemmer
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import models.MulticlassClassifiers as mc_classifiers
import models.MulticlassEvaluators as mc_evluators
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pickle
import MySQLdb
from textblob import TextBlob

import models.CrossValidationEvaluation as CVEval
from sklearn.externals import joblib

class Model:
    def __init__(self):
        self.tfidf_vectorizer = ''
        self.tfidf_vectorizer_3 = ''
        self.test_texts = ''

    def load_data(self, filename, labels):
        dataset = pd.read_excel(filename)
        train_texts = dataset['text'].tolist()
        orig_labels = dataset['label']
        train_labels = dataset['label'].apply(labels.index).tolist()

        return dataset, train_texts, train_labels, orig_labels

    def write_to_excel(self, text, labels, filename):
        df = pd.DataFrame({'label': labels, 'text': text})
        writer = pd.ExcelWriter(filename, engine='xlsxwriter')
        df.to_excel(writer, sheet_name='Sheet1')
        writer.save()

    def preprocess(self, train_texts):
        normalizer = Normalizer.Normalizer()
        return normalizer.clean_text(train_texts)

    def print_top_n_word(self, dataset, labels, n):
        top_n = []
        for label in labels:
            text = dataset[dataset['label'] == label]['text']
            #text = ' '.join(text)
            text = ' '.join(str(v) for v in text)
            text = text.split()
            print(label, 'Top ', n, ' words:')
            for word in collections.Counter(text).most_common(n):
                print(word)
            #top_n.append(collections.Counter(text).most_common(n))
        #pd.options.display.max_colwidth = 500
        #df = pd.DataFrame({'Most Common': top_n}, index = labels)
        #print(df)

    def stemming_tokenizer(self, text):
        stemmer = Porter2Stemmer()
        return [stemmer.stem(w) for w in word_tokenize(text)]

    def feature_extraction(self, train_texts, ngram_range):
        '''
        TfidfVectorizer is equivalent to CountVectorizer followed by TfidfTransformer.
        '''

        tfidf_vectorizer = TfidfVectorizer(tokenizer=self.stemming_tokenizer, ngram_range=ngram_range)
        train_texts_tfidf = tfidf_vectorizer.fit_transform(train_texts)

        return train_texts_tfidf, tfidf_vectorizer
        # print(train_texts_tfidf)                          #-> the dataset with numerical features
        # print(count_vect.get_stop_words())                #-> default stop words

    def get_most_common_terms(self, train_texts, n_common_terms):
        count_vectorizer = CountVectorizer()
        train_texts_count = count_vectorizer.fit_transform(train_texts)

        occ = np.asarray(train_texts_count.sum(axis=0)).ravel().tolist()
        counts_df = pd.DataFrame({'term': count_vectorizer.get_feature_names(), 'occurrences': occ})
        print(counts_df.sort_values(by='occurrences', ascending=False).head(n_common_terms))

        return count_vectorizer

    def get_most_common_terms_avg_tfidf(self, train_texts_tfidf, tfidf_vectorizer, n_common_terms):
        weights = np.asarray(train_texts_tfidf.mean(axis=0)).ravel().tolist()
        counts_df = pd.DataFrame({'term': tfidf_vectorizer.get_feature_names(), 'weight': weights})
        print(counts_df.sort_values(by='weight', ascending=False).head(20))

    def save_vectorizers(self, tfidf_vectorizer, filename):
        pickle.dump(tfidf_vectorizer.vocabulary_, open(filename + ".pkl", "wb"))
        print(filename, ' vectorizer saved successfully!')

    def create_models(self, train_texts_tfidf, train_labels):
        '''
        Only call if there's a new unsaved model.
        Models:
            ngrams (1,2)    SAVED
            ngrams (1,3)    SAVED

        '''

        classifier = mc_classifiers.Classififers(train_texts_tfidf, train_labels)
        gradient_boosting_classifier = classifier.gradient_boosting_classifier()
        # gaussian_process_classifier = classifier.gaussian_process_classifier().predict(test_texts_tfidf)
        linear_svc = classifier.linear_svc()
        logistic_regression = classifier.logistic_regression()
        logistic_regression_cv = classifier.logistic_regression_cv()
        sgdc_classifier = classifier.sgdc_classifier()
        perceptron = classifier.perceptron()
        passive_agressive_classifer = classifier.passive_agressive_classifer()

        print('Models successfully saved into disk!')

    def machine_learning(self, test_texts_tfidf):
        gradient_boosting_classifier = joblib.load("models/gbc_ngrams2_model.pkl").predict(test_texts_tfidf)
        #gaussian_process_classifier = classifier.gaussian_process_classifier().predict(test_texts_tfidf)
        linear_svc = joblib.load("models/lsvc_ngrams2_model.pkl").predict(test_texts_tfidf)
        logistic_regression = joblib.load("models/lr_ngrams2_model.pkl").predict(test_texts_tfidf)
        logistic_regression_cv = joblib.load("models/lrcv_ngrams2_model.pkl").predict(test_texts_tfidf)
        sgdc_classifier = joblib.load("models/sgdc_ngrams2_model.pkl").predict(test_texts_tfidf)
        perceptron = joblib.load("models/p_ngrams2_model.pkl").predict(test_texts_tfidf)
        passive_agressive_classifer = joblib.load("models/pac_ngrams2_model.pkl").predict(test_texts_tfidf)

        model_predictions = {'GBC': gradient_boosting_classifier,
                #'GPC': gaussian_process_classifier,
                'LSVC': linear_svc,
                'LR': logistic_regression,
                'LRCV': logistic_regression_cv,
                'SGDCC': sgdc_classifier,
                'P': perceptron,
                'PAC': passive_agressive_classifer}

        return model_predictions

    def cv_eval(self, train_texts_tfidf, train_labels):
        CVEval.CrossValidation(train_texts_tfidf, train_labels).performCV()
    # def setTargetDB(self, targetDB):
    #     '''
    #     Configure variable db based on persona of the person.
    #     1 db per person.
    #
    #     DB: testingdb_sportsfan,
    #         testingdb_fangirl,
    #         testingdb_gamer,
    #         testingdb_foodie,
    #         testingdb_melancholic,
    #     '''
    #
    #     conn = MySQLdb.connect(host="localhost", user="root", passwd="root", db=targetDB, use_unicode=True,
    #                            charset="utf8mb4")
    #
    #     return conn

    # def load_db(self, targetDB):
    #     conn = self.setTargetDB(targetDB)  # change accordingly
    #     cursor_posts = conn.cursor()
    #     cursor_likes = conn.cursor()
    #     cursor_events = conn.cursor()
    #
    #     cursor_posts.execute("""
    #                     SELECT post
    #                     FROM testingposts;
    #                     """)
    #
    #     self.test_texts = ([i[0] for i in cursor_posts.fetchall()])
    #     cursor_likes.execute("""
    #                     SELECT liked_page
    #                     FROM testinglikes;
    #                     """)
    #     self.test_texts += [i[0] for i in cursor_likes.fetchall()]
    #     cursor_events.execute("""
    #                     SELECT event
    #                     FROM testingevents;
    #                     """)
    #     self.test_texts += [i[0] for i in cursor_events.fetchall()]
    #     print(len(self.test_texts))
    #     print(self.test_texts)

    def run(self):
        labels = ['No Label', 'The Fangirl/Fanboy', 'The Foodie', 'The Gamer', 'The Melancholic', 'The Sports Fanatic']
        training_dataset_dir = "Training Dataset/"
        training_dataset_filename = training_dataset_dir + "(MERGED) Training.xlsx"
        preprocessed_filename = training_dataset_dir + "Preprocessed.xlsx"

        ngram_range_2 = (1, 2)
        ngram_range_3 = (1, 3)
        n_common_terms = 20
        n_top_words_per_label = 10

        # 1.) Data Preparation
        dataset, train_texts, train_labels, orig_labels = self.load_data(training_dataset_filename, labels)

        # print(dataset)
        print(len(train_texts))
        # print(train_texts)
        # print(train_labels)

        # 2.) Pre-processing of Text
        print("------PRE-PROCESSED------")
        train_texts = self.preprocess(train_texts)

        # Print top 10 words for each label using the pre-processed dataset
        self.write_to_excel(train_texts, orig_labels, preprocessed_filename)
        preprocessed_dataset = pd.read_excel(preprocessed_filename)
        # self.print_top_n_word(preprocessed_dataset, labels, n_top_words_per_label)

        # 3.) Feature Extraction
        train_texts_tfidf, tfidf_vectorizer = self.feature_extraction(train_texts, ngram_range_2)
        print('TFIDF Shape of 1-2 Ngrams: ', train_texts_tfidf.shape)
        print('1-2 ngrams features')
        #print(tfidf_vectorizer.get_feature_names())

        train_texts_tfidf_3, tfidf_vectorizer_3 = self.feature_extraction(train_texts, ngram_range_3)
        print('TFIDF Shape of 1-3 Ngrams: ', train_texts_tfidf_3.shape)
        print('1-3 ngrams features')
        #print(tfidf_vectorizer_3.get_feature_names())

        #Save Vectorizer
        # self.save_vectorizers(tfidf_vectorizer, "models/tfidf_vectorizer_2")
        # self.save_vectorizers(tfidf_vectorizer_3, "models/tfidf_vectorizer_3")

        # print('TFIDF Features', tfidf_vectorizer.get_feature_names())
        # print(train_texts_tfidf)                              #-> the dataset with numerical features
        # print(count_vect.get_stop_words())                    #-> default stop words

        # self.get_most_common_terms(train_texts, n_common_terms)
        # self.get_most_common_terms_avg_tfidf(train_texts_tfidf, tfidf_vectorizer, n_common_terms)

        # 4. Training
        '''
        WARNING: Comment out only if there's new models.
        self.create_models(train_texts_tfidf_3, train_labels)
        '''

        # test_texts = ['craving for Mcdo fries', 'Join kayo dota.', 'Grabe naman.', 'I love NBA!', 'Fan ako ni Daniel!!!']
        #
        # test_texts_tfidf = tfidf_vectorizer.transform(self.test_texts)

        # model_predictions = self.machine_learning(test_texts_tfidf)
        # print('Gradient Boosting Classifer', model_predictions['GBC'])
        # print('Linear SVC', model_predictions['LSVC'])
        # print('Logistic Regression',  model_predictions['LR'])
        # print('Logistic Regression CV',  model_predictions['LRCV'])
        # print('SGDC Classifier',  model_predictions['SGDCC'])
        # print('Perceptron',  model_predictions['P'])
        # print('Passive Aggressive Classifier',  model_predictions['PAC'])

        # 5. Evaluate Models
        # evaluator = mc_evluators.Evaluators([0, 1], predicted2)
        # print(evaluator.cohen_kappa_score())
        # evaluate_model([3,1], model_predictons)

        # classifier = mc_classifiers.Classififers(train_texts_tfidf, train_labels)

        # print('---Ngrams range: 1-2')
        # self.cv_eval(train_texts_tfidf, train_labels)
        print('---Ngrams range: 1-3')
        self.cv_eval(train_texts_tfidf_3, train_labels)

        print('END')