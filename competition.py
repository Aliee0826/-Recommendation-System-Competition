"""
Method Description:
Features:
1. business features: 
    numeric/binary features (review count, is_open, ..)
    category embedding -> use a word embedding model SIF to embed categories
2. user-business interaction features: 
    build a dnn model to impute unknown user-business pairs' [useful_review, funny_review, cool_review, likes]
3. user features:
    one-hot encoding on users' review features (useful, compliment, ..)
    other numeric features (number of friends, number of years as elite, ..)

Model:
DNN model to predict user-business interaction features
CatBoost model to predict final rating stars


Error Distribution:
{'>=0 and <1:': 101950,
 '>=1 and <2:': 33184,
 '>=2 and <3:': 6175,
 '>=3 and <4:': 735,
 '>=4:': 0}


 RMSE:
 0.978843959383503


 Execution Time:
 285s
"""



import sys
import csv
from datetime import datetime
from pyspark import SparkContext, SparkConf
import json
import pickle
import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec, KeyedVectors
from sklearn.decomposition import PCA

from tensorflow.keras.models import load_model
from tensorflow.keras.models import save_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


# 1. generate features
# 1.1 extract features
def encoding_photo_label(label_list):
    # initialize ['food', 'drink', 'outside', 'inside', 'menu']
    labels = ['food', 'drink', 'outside', 'inside', 'menu']
    ls = [0, 0, 0, 0, 0]
    for i in range(len(labels)):
        if labels[i] in label_list:
            ls[i] = 1
    return ls


def get_features(folder_path):
    # get features
    # business features
    business = sc.textFile(folder_path + '/business.json').map(lambda line: json.loads(line))\
                                                          .map(lambda x: [x['business_id'], x['stars'], x['review_count'], x['is_open'], x['categories'], 
                                                                          x['state'], x['city'], x['latitude'], x['longitude'], x['attributes']])
    checkin = sc.textFile(folder_path + '/checkin.json').map(lambda line: json.loads(line))\
                                                        .map(lambda x: [x['business_id'], sum(x['time'].values()), sum(x['time'].values())/len(x['time'])])
    photo = sc.textFile(folder_path + '/photo.json').map(lambda line: json.loads(line))\
                                                    .map(lambda x: (x['business_id'], x['label'])).groupByKey().mapValues(lambda x: encoding_photo_label(set(x)))\
                                                    .map(lambda x: [x[0], x[1][0], x[1][1], x[1][2], x[1][3], x[1][4]])
    business = pd.DataFrame(business.collect(), columns = ['business_id', 'stars', 'review_count', 'is_open', 'categories', 
                                            'state', 'city', 'latitude', 'longitude', 'attributes'])
    checkin = pd.DataFrame(checkin.collect(), columns = ['business_id', 'total_checkin', 'avg_checkin'])
    photo = pd.DataFrame(photo.collect(), columns = ['business_id', 'food', 'drink', 'outside', 'inside', 'menu'])           
    busniess_feature = business.merge(checkin, how="outer", on='business_id')
    busniess_feature = busniess_feature.merge(photo, how="outer", on='business_id')

    # user features
    user = sc.textFile(folder_path + '/user.json').map(lambda x: json.loads(x))\
                                                  .map(lambda x: [x['user_id'], x['review_count'], x['average_stars'],
                                                                0 if x['friends'] == 'None' else len(x['friends']),
                                                                0 if x['elite'] == 'None' else len(x["elite"]),
                                                                x['useful'], x['funny'], x['cool'], x['fans'],
                                                                x['compliment_hot'], x['compliment_more'], x['compliment_profile'], x['compliment_cute'], 
                                                                x['compliment_list'], x['compliment_note'], x['compliment_plain'], x['compliment_cool'], 
                                                                x['compliment_funny'], x['compliment_writer'], x['compliment_photos']])
    user_feature = pd.DataFrame(user.collect(), columns = ['user_id', 'review_count', 'average_stars', 'num_friends', "num_elite",
                                                                'useful_user', 'funny_user', 'cool_user','fans',
                                                                'compliment_hot', 'compliment_more','compliment_profile','compliment_cute',
                                                                'compliment_list','compliment_note','compliment_plain','compliment_cool',
                                                                'compliment_funny','compliment_writer','compliment_photos'])
    
    # user - business features
    review = sc.textFile(folder_path + '/review_train.json').map(lambda x: json.loads(x))\
                                                      .map(lambda x: [x['user_id'], x['business_id'], x['useful'], x['funny'], x['cool'], x['text']])
    tip = sc.textFile(folder_path + '/tip.json').map(lambda line: json.loads(line))\
                                                .map(lambda x: [x['user_id'], x['business_id'], x['text'], x['likes']])
    review = pd.DataFrame(review.collect(), columns = ['user_id', 'business_id', 'useful_review', 'funny_review', 'cool_review', 'review'])
    tip = pd.DataFrame(tip.collect(), columns = ['user_id', 'business_id', 'tip', 'likes'])
    user_business_feature = review.merge(tip, how="outer", on=['user_id', 'business_id'])
    
    return busniess_feature, user_feature, user_business_feature



# 1.2 business category embedding
class SIF():
    """Class of SIF embedding for contents' tags
    """
    
    def __init__(self, tags: pd.Series, max_len):
        self.tags_list = tags.tolist()
        self.max_len = max_len


    def _train_word2vec(self, tags_list=None):
        if tags_list is None:
            tags_list = self.tags_list
        # set up the model
        w2v_model = Word2Vec(vector_size=self.max_len, 
                            window=2, 
                            min_count=3,
                            negative=3,
                            alpha=0.03)
        # build vocab
        print("Build Vocab")
        w2v_model.build_vocab(tags_list, progress_per=20000)
        print("--> Vocab Size: " + str(len(w2v_model.wv.index_to_key)))
        # train w2v model
        print("Train Model")
        w2v_model.train(tags_list, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
        self.w2v_model = w2v_model
        self.w2v_vec = w2v_model.wv
        print("Finish Training")
        return w2v_model
        

    def _get_word_vec(self, w2v_vec, word):
        """
        given a word, return the word vector in w2v model
        if the word is not in the vocab, return [0]*embedding size
        w2v_vec: Word2Vec model
        word(__string__): a word 
        """
        try:
            return w2v_vec.get_vector(word)
        except KeyError:
            return np.zeros(w2v_vec.vector_size)


    def _get_word_weight(self, w2v_vec, word, alpha):
        """
        given a word, return the weight word in w2v model
        if the word is not in the vocab, return 0
        formula: alpha/(alpha+frequency)

        w2v_vec: Word2Vec model
        word(__string__): a word 
        alpha(__float__): hyperparameter
        """
        try:
            ct = w2v_vec.get_vecattr(word, "count") 
        except KeyError:
            ct = 0
        freq = ct / len(w2v_vec.key_to_index)
        return alpha / (alpha + freq)


    def _get_weighted_avg(self, sentence, alpha, embedding_size, w2v_vec=None):
        """
        given a list of sentences and the word vectors
        uss the weighted sum of words in each sentence to represent the sentence
        formula: [sum(weight*word_vec)/number_of_sentence for word_vec in sentence]

        sentence_list: a list of sentences to be embedded
        w2v_vec: Word2Vec model
        word(__string__): a word 
        alpha(__float__): hyperparameter
        embedding_size: size of word vectors
        sample_size: number of sentences
        """
        if w2v_vec is None:
            try:
                w2v_vec = self.w2v_vec
            except:
                w2v_model = self._train_word2vec(self.tags_list)
                w2v_vec = w2v_model.wv

        sent_vec = np.zeros(embedding_size)
        if sent_vec != []:
            for w in sentence:
                wt = self._get_word_weight(w2v_vec, w, alpha)
                vec_w = self._get_word_vec(w2v_vec, w)
                sent_vec += wt * vec_w
        return sent_vec.tolist()


    def _PCA_revise(self, sent_vec, u):
        # get sentence vectors, vs = vs - (u x uT) x vs
        sub = np.multiply(u, np.array(sent_vec))
        sent_vec_new = np.subtract(np.array(sent_vec), sub)
        return list(sent_vec_new)


    def SIF(self, w2v_vec=None, alpha=1e-3):
        if w2v_vec is None:
            try:
                w2v_vec = self.w2v_vec
            except:
                w2v_model = self._train_word2vec(self.tags_list)
                w2v_vec = w2v_model.wv

        sample_size = len(self.tags_list)
        embedding_size = w2v_vec.vector_size
        sent_ = pd.Series(self.tags_list, name = 'tags')
        print("Calculate weighted sum of sentences")
        sent_matrix = sent_.apply(lambda x: self._get_weighted_avg(x,
                                                                alpha, 
                                                                embedding_size,
                                                                w2v_vec
                                                                ))
        print("Revise sentence matrix by PCA")
        print("--> Calculate first component of PCA")
        # calculate PCA of this sentence set
        pca = PCA(n_components=embedding_size)
        array_sent_matrix = np.array(sent_matrix.tolist())
        pca.fit(array_sent_matrix)
        u = pca.components_[0]  # the PCA vector
        u = np.multiply(u, np.transpose(u))  # u x uT
        # pad the vector  (occurs if we have less sentences than embeddings_size)
        print("--> Pad u if shorter than embedding size")
        if len(u) < embedding_size:
            for i in range(embedding_size - len(u)):
                u = np.append(u, 0)  # add needed extension for multiplication below
        # get sentence vectors, vs = vs - (u x uT) x vs
        print("--> Revise sentence matrix")
        sent_matrix = pd.Series(
            sent_matrix.tolist(), 
            name='embed'
            ).apply(lambda x: self._PCA_revise(x, u))
        print("Embedding Finished!")
        return sent_matrix.tolist()

def get_categories(category_str):
    if category_str is None:
        return []
    else:
        ls = category_str.split(',')
        return [x.replace(' ', '') for x in ls]

def business_embedding(business_feature, embed_size=50):
    cate = business_feature.categories.apply(lambda x: get_categories(x))
    embedding = SIF(cate, embed_size)
    sif_matrix = embedding.SIF()
    df_sif_matrix = pd.DataFrame(sif_matrix)
    df_sif_matrix.columns = ['category_{}'.format(i) for i in range(len(df_sif_matrix.columns))]
    business_feature_embedding = business_feature.drop(columns=['city', 'state', 'attributes', 'categories', 'attributes'])
    business_feature_embedding = pd.concat([business_feature_embedding, df_sif_matrix], axis=1)
    return business_feature_embedding, embedding


# 1.3 user-business interaction embedding
def create_model(n_inputs, n_outputs, learning_rate):
    """Create the model.

        Returns:
            ColdStartDNN(): The class instance.
        """
    model = Sequential()
    model.add(BatchNormalization())
    model.add(Dense(
        256,
        input_dim=n_inputs,
        activation='relu',
        name='deep_1'))
    model.add(Dropout(0.5))
    model.add(Dense(
        128,
        activation='relu',
        name='deep_2'))
    model.add(Dropout(0.5))
    model.add(Dense(
        64,
        activation='relu',
        name='deep_3'))
    model.add(Dense(
        n_outputs,
        activation='linear',
        name='deep_out'))
    model.compile(loss='mean_squared_error',
                  optimizer=Adam(lr=learning_rate),
                  metrics=["mse"])

    return model


def fit_model(X, y, verbose='auto', epochs=1, model=None):
    """Fit the model.

        Returns:
            ColdStartDNN(): The class instance.
      """
    if model is None:
        raise Exception('Create model first')

    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    callbacks = [early_stopping]

    model.fit(X, y, verbose=verbose,\
                  epochs=epochs, batch_size=1024,\
                  validation_split=0.2, shuffle=True,\
                  callbacks=callbacks)

    return model



def get_u_b_embedding(val, business_feature, user_feature, u_b_num, embed_model):
    # get X, y
    if len(val.columns) == 3:
        val.columns = ['user_id', 'business_id', 'true_star']
        val = val.merge(business_feature, on='business_id', how='left')
        val = val.merge(user_feature, on='user_id', how='left')
        Xval = val.drop(columns=['true_star', 'user_id', 'business_id'])
        yval = val.true_star
    else:
        val = val.merge(business_feature, on='business_id', how='left')
        val = val.merge(user_feature, on='user_id', how='left')
        Xval = val.drop(columns=['user_id', 'business_id'])

    # get embedding
    X_input = np.array(Xval.fillna(0))
    ub_embed = embed_model.predict(X_input, batch_size=1024)
    pred_num = pd.DataFrame(ub_embed, columns=list(u_b_num.columns)[2:])
    Xval = pd.concat([pred_num, Xval], axis=1)
    return Xval


def train_embedding():
    ub_num = pd.read_csv('Competition/code/cache/u_b_num.csv', index_col=0)
    X_for_num = ub_num[['user_id', 'business_id']].merge(business_feature, on=['business_id'], how='left')
    X_for_num = X_for_num.merge(user_feature, on=['user_id'], how='left')
    dnn_final = create_model(81, 4, 1e-3)
    X = np.array(X_for_num.drop(columns=['user_id', 'business_id']).fillna(0))
    y = np.array(ub_num.drop(columns=['user_id', 'business_id']))
    dnn_final = fit_model(X, y, epochs=50, model=dnn_final)
    save_model(dnn_final, 'Competition/code/cache/dnn_embed.h5')
    return dnn_final


def get_prediction(model, X, val_file):
    y_pred = model.predict(X)

    result = val_file[['user_id', 'business_id']].copy()
    result['prediction']=y_pred.tolist()

    result.loc[result.prediction>5, 'prediction'] = 5
    result.loc[result.prediction<1, 'prediction'] = 1

    return result.values.tolist()


# write output file
def write_ouput_file(output_file_path, result):
    with open(output_file_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['user_id', 'business_id', 'prediction'])
        writer.writerows(result)



# ====================================================================== output ======================================================================================

if __name__ == "__main__":
    # input
    
    input_ls = {'folder_path': 'Competition/asnlib/publicdata/data',
                'test_file_path': 'Competition/asnlib/publicdata/data/yelp_val.csv',
                'Output_file_path': 'Competition/asnlib/publicdata/output/output.csv'}
    '''
    input_ls = {'folder_path': sys.argv[1],
                'test_file_path': sys.argv[2],
                'Output_file_path': sys.argv[3],}
    '''
   
    folder_path = input_ls['folder_path']
    output_file_path = input_ls['Output_file_path']
    test_file_path = input_ls['test_file_path']

    conf = SparkConf().setAppName("data").setMaster('local[*]')
    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")
    conf.set("spark.executor.memory", "4g")
    conf.set("spark.driver.memory", "4g")

    # output
    start = datetime.now()
    # models
    # dnn_model = load_model('cache/dnn_embed_ubnum.h5', compile = False)
    with open("Competition/code/cache/catboost_plus_ubnum_81.pkl", 'rb') as file:
        catboost_model = pickle.load(file)

    # data
    train = pd.read_csv(folder_path + '/yelp_train.csv')
    val = pd.read_csv(test_file_path)
    business_feature = pd.read_csv('Competition/code/cache/business_feature.csv', index_col=0)
    user_feature = pd.read_csv('Competition/code/cache/usesr_feature.csv', index_col=0)
    ub_num = pd.read_csv('Competition/code/cache/ub_num_embedding.csv', index_col=0)
    # user_business_feature = pd.read_feather('cache/usesr_business_feature.feather')

    # embedding
    dnn_model = train_embedding()
    Xval = get_u_b_embedding(val, business_feature, user_feature, ub_num, dnn_model)

    # predict
    result = get_prediction(catboost_model, Xval, val)
    
    write_ouput_file(output_file_path, result)
    end = datetime.now()
    print((end - start).seconds)
