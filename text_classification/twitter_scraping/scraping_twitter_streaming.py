import os
import pytz
import pickle
import logging
import datetime
import time
import traceback

from requests_oauthlib import OAuth1Session
import tweepy

#dict型でCONSUMER_KEYなどのtwitterAPIを使用する自分のキーを渡しました。
import twitter_scraping.twitter_access as twitter_access

TWITTER_BASE_DIR = 'scraping_pickle_dir'


formatter = '[%(levelname)s] %(asctime)s %(message)s %(name)s'
logging.basicConfig(level=logging.INFO, format=formatter)
logger = logging.getLogger(__name__)

CONSUMER_KEY = twitter_access.twitter_dict['CONSUMER_KEY']
CONSUMER_SECRET =twitter_access.twitter_dict['CONSUMER_SECRET']
ACCESS_TOKEN = twitter_access.twitter_dict['ACCESS_TOKEN']
ACCESS_TOKEN_SECRET = twitter_access.twitter_dict['ACCESS_TOKEN_SECRET']


# 認証情報を設定する。
auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)


class MyStreamListener(tweepy.StreamListener):

    def __init__(self):
        self.tweets = []
        self.count = 0
        self.time_now = datetime.datetime.now() 
        # self.time_later = self.time_now + datetime.timedelta(minutes=15)
        self.time_threshold=datetime.datetime.strptime(self.get_quarter_time(self.time_now), '%Y-%m-%d %H:%M:%S')
        super().__init__()

    def on_status(self, status):
        # logger every 100 tweets
        if self.count % 100 == 0:
            logger.info("count : {}".format(self.count))

        tweets_dict = {
            'id': status.id,
            'created_at': status.created_at,
            'screen_name':  status.author.screen_name,
            'text': status.text,
            'retweet_count': status.retweet_count,
            'name': status.author.name,
        }

        self.tweets.append(tweets_dict)
        self.count += 1

        if datetime.datetime.now() > self.time_threshold:
            # trans utc
            time_now_utc = self.time_now - datetime.timedelta(hours=9)
            time_threshold_utc = self.time_threshold - datetime.timedelta(hours=9)
            filename = time_now_utc.strftime('%Y%m%d%H_%M')+'_'+str(time_threshold_utc.minute)+'.pickle'
            filename_path = os.path.join(TWITTER_BASE_DIR, filename)
            self.save_tweets(filename_path,self.tweets)
            
            # 初期化
            self.__init__()

    #始めた時間から15分ずつだとキリが悪いから、
    # 0~15,15~30,30~45,45~60分の間でscrapingできるようにするための規則を示したメソッド
    def get_quarter_time(self,_time_now):
        minute_quarter = _time_now.minute
        quarter_time = ((minute_quarter//15)+1)*15
        if not quarter_time == 60:
            return _time_now.strftime('%Y-%m-%d %H:{}:00').format(quarter_time)
        else:
            _time_now_hour = _time_now+datetime.timedelta(hours=1)
            return _time_now_hour.strftime('%Y-%m-%d %H:00:00').format(_time_now_hour)
            
    def save_tweets(self,filename, tweets):
        with open(filename, 'wb') as file:
            pickle.dump(tweets, file)
            logger.info("create_file : {}".format(filename))


if __name__ == '__main__':
    while True:
        try:
            # 認証情報とStreamListenerを指定してStreamオブジェクトを取得
            stream = tweepy.Stream(auth, MyStreamListener())
            stream.filter(languages=['en'],track=['BITCOIN','Bitcoin','bitcoin','BTC/USD'])
        except:
            logger.warning("========ERROR=========")
            traceback.print_exc()
            time.sleep(5)
            continue

