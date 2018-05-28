import pandas as pd
import MySQLdb
import utils.Normalizer as Normalizer
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
import pickle
import operator

class PersonaIdentification:
    def __init__(self):
        self.test_posts = ''
        self.test_likes = ''
        self.test_events = ''
        self.models_dir = 'models/'
        self.bestmodels = []
        self.labels = ['No Label', 'The Fangirl/Fanboy', 'The Foodie', 'The Gamer', 'The Melancholic', 'The Sports Fanatic']


    def initModels(self):
        self.bestmodels.append(self.models_dir + 'gbc_ngrams2_model.pkl')
        self.bestmodels.append(self.models_dir + 'gbc_ngrams3_model.pkl')
        self.bestmodels.append(self.models_dir + 'lsvc_ngrams2_model.pkl')              #highest F1 score
        self.bestmodels.append(self.models_dir + 'lsvc_ngrams3_model.pkl')
        self.bestmodels.append(self.models_dir + 'lr_ngrams2_model.pkl')
        self.bestmodels.append(self.models_dir + 'lr_ngrams3_model.pkl')
        self.bestmodels.append(self.models_dir + 'lrcv_ngrams2_model.pkl' )             #one of the best model
        self.bestmodels.append(self.models_dir + 'lrcv_ngrams3_model.pkl')
        self.bestmodels.append(self.models_dir + 'sgdc_ngrams2_model.pkl')
        self.bestmodels.append(self.models_dir + 'sgdc_ngrams3_model.pkl')              # highest accuracy
        self.bestmodels.append(self.models_dir + 'p_ngrams2_model.pkl')
        self.bestmodels.append(self.models_dir + 'p_ngrams3_model.pkl')
        self.bestmodels.append(self.models_dir + 'pac_ngrams2_model.pkl')
        self.bestmodels.append(self.models_dir + 'pac_ngrams3_model.pkl')

    def setTargetDB(self, targetDB):
        '''
        Configure variable db based on persona of the person.
        1 db per person.

        DB: testingdb_sportsfan,
            testingdb_fangirl,
            testingdb_gamer,
            testingdb_foodie,
            testingdb_melancholic,
        '''

        conn = MySQLdb.connect(host="localhost", user="root", passwd="root", db=targetDB, use_unicode=True,
                               charset="utf8mb4")

        return conn

    def load_data(self, targetDB):
        conn = self.setTargetDB(targetDB)  # change accordingly
        cursor_posts = conn.cursor()
        cursor_likes = conn.cursor()
        cursor_events = conn.cursor()

        cursor_posts.execute("""
                        SELECT post
                        FROM testingposts;
                        """)

        self.test_posts = ([i[0] for i in cursor_posts.fetchall()])
        cursor_likes.execute("""
                        SELECT liked_page
                        FROM testinglikes;
                        """)
        self.test_likes = [i[0] for i in cursor_likes.fetchall()]
        cursor_events.execute("""
                        SELECT event
                        FROM testingevents;
                        """)
        self.test_events = [i[0] for i in cursor_events.fetchall()]
        # print(len(self.test_texts))
        # print(self.test_texts)

    def preprocess(self, train_texts):
        normalizer = Normalizer.Normalizer()
        return normalizer.clean_text(train_texts)

    def machine_learning(self, model_filename, test_texts):

        if(model_filename.find("ngrams3")):
            vectorizer = TfidfVectorizer(vocabulary=pickle.load(open("models/tfidf_vectorizer_3.pkl", "rb")))
        else:
            vectorizer = TfidfVectorizer(vocabulary=pickle.load(open("models/tfidf_vectorizer_2.pkl", "rb")))

        model = joblib.load(model_filename).predict(vectorizer.fit_transform(test_texts))
        model_predictions = {model_filename: model}
        return model_predictions

    def str_join(*args):
        return ''.join(map(str, args))

    def append_labels(self, conn, table, num_labels):

        for i, num_label in enumerate(num_labels):
            cursor = conn.cursor()
            print('Num_Label', num_label)
            cursor.execute('''
                    UPDATE %s 
                    SET label = '%s'
                    WHERE id = %s;
                ''' % (table, self.labels[num_label], i+1))
            conn.commit()

    def get_actual_labels(self,  filename):
        test_actual_labels = []
        dataset = pd.read_excel(filename, sheet_name="Posts")
        test_actual_labels.extend(dataset['label'].apply(self.labels.index).tolist())
        dataset = pd.read_excel(filename, sheet_name="Likes")
        test_actual_labels.extend(dataset['label'].apply(self.labels.index).tolist())
        dataset = pd.read_excel(filename, sheet_name="Events")
        test_actual_labels.extend(dataset['label'].apply(self.labels.index).tolist())

        return test_actual_labels

    def evaluate(self, actual_labels,predicted_labels):
        print(classification_report(actual_labels,predicted_labels))

        print("Accuracy Rate:", accuracy_score(actual_labels, predicted_labels))

    def identifyPersona(self, targetDB):
        conn = self.setTargetDB(targetDB)
        cursor_posts = conn.cursor()
        cursor_likes = conn.cursor()
        cursor_events = conn.cursor()

        cursor_posts.execute("""
                        SELECT label, COUNT(*) AS num 
                        FROM testingposts
                        WHERE label NOT LIKE 'No Label'
                        GROUP BY label
                        ORDER BY num DESC        
                        """)
        cursor_likes.execute("""
                        SELECT label, COUNT(*) AS num 
                        FROM testinglikes
                        WHERE label NOT LIKE 'No Label'
                        GROUP BY label
                        ORDER BY num DESC        
                        """)
        cursor_events.execute("""
                        SELECT label, COUNT(*) AS num 
                        FROM testingevents
                        WHERE label NOT LIKE 'No Label'
                        GROUP BY label
                        ORDER BY num DESC        
                        """)

        posts = dict(cursor_posts.fetchall())
        print(posts)

        likes = dict(cursor_likes.fetchall())
        print(likes)

        events = dict(cursor_events.fetchall())
        print(events)

        posts_persona = {}
        likes_events_persona = {}
        final_persona = {}

        for key, value in posts.items():
            posts_persona[key] = value * .7

        print('---------')
        print(posts_persona)

        for key, value in likes.items():
            try:
                v = value + events[key]
            except:
                v = value # dummy

            likes_events_persona[key] = v * 0.3

        for key, value in events.items():
            try:
                likes_events_persona[key]
            except:
                likes_events_persona[key] = value * 0.3

        print(likes_events_persona)


        # Combine Scores of Posts & Liked Pages and Events
        for key, value in posts_persona.items():
            try:
                final_persona[key] = value + likes_events_persona[key]
            except:
                final_persona[key] = value

        for key, value in likes_events_persona.items():
            try:
                final_persona[key]
            except:
                final_persona[key] = value


        # Print out all personas found
        print(final_persona)

        # Final Persona
        print(max(final_persona.items(), key=operator.itemgetter(1))[0])

    def run(self, targetDB):
        self.initModels()
        # Load Data
        self.load_data(targetDB)
        conn = self.setTargetDB(targetDB)
        test_predicted_labels = []
        testing_dataset_filename = "Testing Dataset/(FANGIRL) Testing.xlsx"

        #Pre-process
        print('TEXT-BASED POSTS:', len(self.test_posts))


        for model in self.bestmodels:
            test_predicted_labels = []
            print('Model: ', model)

            self.test_posts = self.preprocess(self.test_posts)
            predictions = self.machine_learning(model, self.test_posts)
            # print('Predictions:', predictions[model])
            # self.append_labels(conn, 'testingposts', predictions[model])
            test_predicted_labels.extend(predictions[model])

            # print('LIKED PAGES:', len(self.test_likes))
            self.test_likes = self.preprocess(self.test_likes)
            predictions = self.machine_learning(model, self.test_likes)
            # print('Predictions:', predictions[model])
            # self.append_labels(conn, 'testinglikes', predictions[model])
            test_predicted_labels.extend(predictions[model])

            print('EVENTS:', len(self.test_events))
            self.test_events = self.preprocess(self.test_events)
            predictions = self.machine_learning(model, self.test_events)
            # print('Predictions:', predictions[model])
            # self.append_labels(conn, 'testingevents', predictions[model])
            test_predicted_labels.extend(predictions[model])

            print('Test texts: ', len(test_predicted_labels ))
            self.evaluate(self.get_actual_labels(testing_dataset_filename), test_predicted_labels)


