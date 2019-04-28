import glob
import json
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import pprint as pp
questions_folder = glob.glob("../Data/Processed/*")
print(questions_folder)
questions_folder = questions_folder

required_fields = ['creation_date', 'negative_score', 'positive_score', 'compound_score', 'neutral_score', 'pos_post', 'neg_post', 'neu_post']
i = 0
data = []
for file in questions_folder:

    with open(file, encoding="utf8") as ff:
        file_data = json.load(ff)
        for x in file_data:
            row = { your_key: x[your_key] for your_key in required_fields }
            # row['pos_post'] = 1 if row['Sentiment_type'] == 'positive' else 0
            # row['neg_post'] = 1 if row['Sentiment_type'] == 'negative' else 0
            # row['neu_post'] = 1 if row['Sentiment_type'] == 'neutral' else 0
        # required_data = [{ your_key: x[your_key] for your_key in required_fields } for x in file_data]
            data.append(row)

data_df = pd.DataFrame(data)

def count_negative(x):
    return lambda x: (x =='negative').count()

def count_positive(x):
    return lambda x: (x =='positive').count()

def count_neutral(x):
    return lambda x: (x =='neutral').count()

def f(x):
    d = {}

    d['mean_neg_score'] = x['negative_score'].mean()
    d['max_neg_score'] = x['negative_score'].max()

    d['mean_pos_score'] = x['positive_score'].mean()
    d['max_pos_score'] = x['positive_score'].max()

    d['mean_compound_score'] = x['compound_score'].mean()
    d['max_compound_score'] = x['compound_score'].max()

    d['mean_neu_score'] = x['neutral_score'].mean()
    d['max_neu_score'] = x['neutral_score'].max()

    d['positive_posts'] = x['pos_post'].sum()
    d['negative_posts'] = x['neg_post'].sum()
    d['neutral_posts'] = x['neu_post'].sum()

    return pd.Series(d, index=['mean_neg_score', 'max_neg_score', 'mean_pos_score', 'max_pos_score', 'mean_compound_score', 'max_compound_score', 'mean_neu_score', 'max_neu_score', 'positive_posts', 'negative_posts','neutral_posts'])


cal_data = data_df.groupby('creation_date').apply(f).reset_index()
pp.pprint(len(cal_data))
print(cal_data.columns)
cal_data.to_csv("../Data/ChartData/TimeSeries.csv")