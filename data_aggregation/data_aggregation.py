import os
import datetime
import random
import os.path

import pandas as pd
import numpy as np

class DataAggregation(object):
    def __init__(self,num_shop,num_goods):
        self.num_shop = num_shop
        self.num_goods = num_goods

    def create_dataframe(self,file_path):
        # 日付作成
        start = datetime.datetime(2013, 1, 1)
        end = datetime.datetime(2016, 6, 30)

        date_list = []
        while start < end:
            date_list.append(start.strftime('%Y/%m/%d'))
            start += datetime.timedelta(days=1)

        num_shop_goods = self.num_shop + self.num_goods

        # ID(店鋪&商品)取得
        random_id = random.sample(range(num_shop_goods), num_shop_goods)

        # 店鋪idと商品idが連動しないようにした
        self.id_shop = random_id[:self.num_shop]
        self.id_goods = random_id[self.num_shop:]

        shop_list_data_day = []
        goods_list_data_day = []

        for shop in self.id_shop:
            shopid_data = [shop] * len(self.id_goods)
            shop_list_data_day += shopid_data
            goods_list_data_day += self.id_goods

        # 1日あたり生成されるデータ数
        length_shop_goods_par_day = len(goods_list_data_day)

        day_list_test_data = []
        shop_list_test_data = []
        goods_list_test_data = []

        for day in date_list:
            day_test_data = [day] * length_shop_goods_par_day
            day_list_test_data += day_test_data
            shop_list_test_data += shop_list_data_day
            goods_list_test_data += goods_list_data_day

        # 1の答え
        # 売上高と販売数量は0~1000の一様乱数にした

        main_df = pd.DataFrame({
            'shop_id': shop_list_test_data,
            'goods_id': goods_list_test_data,
            'sale': np.random.rand(len(day_list_test_data)) * 1000,
            'sales_quantity': np.random.rand(len(day_list_test_data)) * 1000
        }, index=pd.to_datetime(day_list_test_data))
        # 保存
        main_df.to_csv(file_path)

        def check_df_size(_getsize_df):
            # 500MB以上あるのか判定
            if _getsize_df > 500:
                print("作成されたcsvは {:.2f}MBなので正常終了".format(_getsize_df))
                return True
            else:
                print("作成されたcsvは {:.2f}MBなので、あと {:.2f}MB足りません".format(_getsize_df, 500 - _getsize_df))
                return False

        # 500MB以上あるかチェック
        getsize_df = os.path.getsize(file_path)*9.536743164e-7
        if check_df_size(getsize_df):
            return main_df
        else:
            raise Exception('capacity lack')

    # 2.1と2.２の回答
    def get_monthly_shop_goods_and_sale(self,_df):
        df_sum_month_group = _df.groupby(pd.Grouper(freq='MS'))

        df_month_shop_list = []
        df_month_goods_list = []

        # monthには毎月の1日が格納されている
        for month in df_sum_month_group.sum().index[:]:
            df_month = df_sum_month_group.get_group(month)

            # shop_id,goodsごとのsumをとる
            df_month_shop_group = df_month.groupby('shop_id').sum()
            df_month_goods_group = df_month.groupby('goods_id').sum()

            df_month_shop_group['date'] = month
            df_month_goods_group['date'] = month

            df_month_shop_list.append(df_month_shop_group.loc[:, ['date', 'sale']])
            df_month_goods_list.append(df_month_goods_group.loc[:, ['date', 'sale']])

        # 2.1と2.２の回答
        df_month_shop = pd.concat(df_month_shop_list).reset_index().loc[:, ['date', 'shop_id', 'sale']]
        df_month_goods = pd.concat(df_month_goods_list).reset_index().loc[:, ['date', 'goods_id', 'sale']]

        return df_month_shop,df_month_goods

    def get_daily_mean_goods(self,_df):
        # 2.3の回答

        # 有効数字10桁に変更
        pd.options.display.precision = 10

        def get_sale_division(_x):
            sale, sales_quantity = _x
            # 有効数字10桁
            return float(format(sale / sales_quantity, '.10f'))

        df_day_group = _df.groupby(pd.Grouper(freq='D'))

        df_goods_mean_list = []
        for day in df_day_group.sum().index[:]:

            df_day_division_list = []
            df_day = df_day_group.get_group(day)
            df_day_shop_group = df_day.groupby('shop_id')
            for shop_id in self.id_shop[:]:
                df_day_shop = df_day_shop_group.get_group(shop_id).copy()
                df_day_shop['sale/sale_quantity'] = df_day_shop.loc[:, ['sale', 'sales_quantity']].apply(
                    get_sale_division, axis=1)
                df_day_division_list.append(df_day_shop)
            df_shop_id_day_mean = pd.concat(df_day_division_list).groupby('goods_id').mean()
            df_shop_id_day_mean['day'] = day
            df_goods_mean_list.append(df_shop_id_day_mean)

        df_goods_mean = pd.concat(df_goods_mean_list).reset_index().loc[:, ['day', 'goods_id', 'sale/sale_quantity']]
        return  df_goods_mean

