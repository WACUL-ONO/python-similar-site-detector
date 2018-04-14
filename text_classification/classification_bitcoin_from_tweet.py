import numpy as np
import pandas as pd
import os
import pickle
import re
import collections

from sklearn.model_selection import \
    train_test_split,\
    cross_val_score,\
    RandomizedSearchCV

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

import xgboost as xgb

plt.rcParams['font.family'] = 'sans-serif'


#Pathの格納先
TEXT_PATH = './scraping_pickle_dir/'
VALUE_PATH = './bitcoin_value/'
STOP_WORD_PATH = './stopwords.txt'


class CalcBitcoinText(object):

    def __init__(self):
        self.tfidfs = None
        self.label_array = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.param = {}

    def created_df(self,num_smooth=30):
        text_name_list = sorted(os.listdir(TEXT_PATH))
        value_path_list = sorted(os.listdir(VALUE_PATH))

        df_main_bitcoin_list = []
        for value_path in value_path_list:
            df_bitcoin = pd.read_csv(os.path.join(VALUE_PATH, value_path))
            df_bitcoin_dropna = df_bitcoin.dropna()
            df_main_bitcoin_list.append(df_bitcoin_dropna)

        df_main_bitcoin = pd.concat(df_main_bitcoin_list)
        df_main_bitcoin = df_main_bitcoin.drop_duplicates().reset_index(drop=True)


        def _trans_name(_x):
            year = _x[:4]
            month = _x[5:7]
            day = _x[8:10]
            hour = _x[11:13]
            minute = _x[14:16]
            if int(minute) == 45:
                next_minute = str(0)
            else:
                next_minute = str(int(minute) + 15)
            trans_name = '{}{}{}{}_{}_{}.pickle'.format(year, month, day, hour, minute, next_minute)
            return trans_name

        # スクレイピングしたデータのファイルがあるかチェックする
        def _confirmation_date(_x):
            if str(_x) in text_name_list:
                return True
            return False

        # 最大と最小
        def _get_mean_bitcoin(_x):
            high, low = _x
            # 有効数字10桁
            return (high + low) / 2

        df_main_bitcoin['data_name'] = df_main_bitcoin['Date'].apply(_trans_name)
        df_main_bitcoin['is_data'] = df_main_bitcoin['data_name'].apply(_confirmation_date)
        df_main_bitcoin['Mean'] = df_main_bitcoin.loc[:, ['High', 'Low']].apply(_get_mean_bitcoin, axis=1)

        # 移動平均
        kernel = np.ones(num_smooth) / num_smooth
        smoothed = np.convolve(df_main_bitcoin['Close'], kernel, mode='same')
        df_main_bitcoin['Close_smooth'] = smoothed

        # 平均の変化量計算
        diff_mean_array = np.array(df_main_bitcoin['Close_smooth'][1:]) - np.array(df_main_bitcoin['Close_smooth'][:-1])

        # ラベル
        label = np.array([0 if diff_mean < 0 else 1 for diff_mean in diff_mean_array])

        # 最初は測れないからとりあえず0にする
        label = np.append(np.array([0]), label)

        df_main_bitcoin['label'] = label

        # 端はうまくスムージングされていないから抜く
        df_main_bitcoin = df_main_bitcoin[num_smooth:-num_smooth]

        df_main_bitcoin_is_data = df_main_bitcoin[df_main_bitcoin.is_data == True]
        return df_main_bitcoin_is_data

    def execute_tfidf(self, _df, threshold=5):

        # 絵文字やURLなどを削除
        def _eliminate_character(_text):
            _text = re.sub(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-…]+', '', _text, )
            _text = re.sub("["
                           u"\U0001F600-\U0001F64F"
                           u"\U0001F300-\U0001F5FF"
                           u"\U0001F680-\U0001F6FF"
                           u"\U0001F1E0-\U0001F1FF"
                           "]+", '', _text, )
            _text = re.sub(r'@[\w]+', '', _text, )
            _text = re.sub('RT|#', "", _text)
            _text = re.sub('\n', " ", _text)
            _text = re.sub(r'[︰-＠]', "", _text)
            _text = re.sub(r'^[!-~]+$', "", _text)
            return _text

        corpus_list = []
        corpus_list_flatten = []
        for file_path in np.array(_df['data_name']):
            dict_data = pickle.load(open(os.path.join(TEXT_PATH, file_path), 'rb'))
            test_corpus = ' '.join([_eliminate_character(_dict['text']) for _dict in dict_data])
            test_corpus = test_corpus.split()
            # test_corpus =[eliminate_character(_dict['text']) for _dict in dict_data]
            corpus_list_flatten += test_corpus
            corpus_list.append(' '.join(test_corpus))

        # ストップワードは以下を使用
        # https://github.com/ravikiranj/twitter-sentiment-analyzer/blob/master/data/feature_list/stopwords.txt
        file = open(STOP_WORD_PATH, 'r')
        word = file.readline()
        stop_word_list = []
        while word:
            word = word.replace('\n', '')
            stop_word_list.append(word)
            word = file.readline()

        corpus_counter = collections.Counter(corpus_list_flatten)
        corpus_counter_sort = sorted(corpus_counter.items(), key=lambda x: x[1], reverse=True)
        corpus_counter_stopword = [word for word, num in corpus_counter.items() if num < threshold]
        stop_word_list += corpus_counter_stopword

        # ベクトル化
        vectorizer = TfidfVectorizer(use_idf=True, stop_words=stop_word_list)
        self.tfidfs = vectorizer.fit_transform(corpus_list)
        self.label_array = np.array(_df['label'])

    def get_model(self,test_size=0.5, random_state=0):
        self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.tfidfs.toarray(), self.label_array,
                                                            test_size=test_size, random_state=random_state)
        if not self.param:
            model = xgb.XGBClassifier()
        else:
            model = xgb.XGBClassifier(**self.param)

        # 学習
        model.fit(self.X_train, self.y_train)

        return model

    def execute_turning_rs(self,_model):

        param_test = {
            'max_depth': [3, 4, 5, 6, 7],
            'min_child_weight': [6, 7, 8, 9, 10],
            'gamma': [i / 10.0 for i in range(0, 5)],
            'subsample': [i / 10.0 for i in range(6, 10)],
            'colsample_bytree': [i / 10.0 for i in range(6, 10)],
            'reg_alpha': [0, 0.05, 0.1, 0.2, 0.3, 0.4],

        }
        rsearch = RandomizedSearchCV(_model, param_test, n_jobs=-1)
        rsearch.fit(self.X_train, self.y_train)
        self.param = rsearch.best_params_

    def evaluate_model(self,_model):
        predicted = _model.predict(self.X_test)
        expected = self.y_test
        print("テストデータによる正解率 : {}".format(metrics.accuracy_score(expected, predicted)))
        confmat = confusion_matrix(y_true=expected, y_pred=predicted)

        score = cross_val_score(estimator=_model,
                                X=self.X_train,
                                y=self.y_train,
                                cv=5,
                                n_jobs=-1)
        print("クロスバリデーションによる検証結果 : %.3f +/- %.3f" % (np.mean(score), np.std(score)))

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(confmat.shape[0]):
            for j in range(confmat.shape[1]):
                ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
        plt.title('confusion_matrix')
        plt.xlabel('predicted label')
        plt.ylabel('true label')

        plt.tight_layout()
        plt.show()

        probas_ = _model.predict_proba(self.X_test)
        fpr, tpr, thresholds = roc_curve(self.y_test, probas_[:, 1])
        roc_auc_area = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(fpr, tpr, color='r', lw=2,
                label='ROC curve (area = {:.2f})'.format(roc_auc_area))
        ax.plot([0, 1], [0, 1], color='b', linestyle='--')
        plt.title('ROC curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.show()








