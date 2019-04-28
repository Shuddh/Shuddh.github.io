import nltk
import re
import glob
import json
import string
from nltk.corpus import stopwords
import pandas as pd
en_stops = set(stopwords.words('english'))
nltk.download("wordnet")
nltk.download("stopwords")
nltk.download('punkt')
import matplotlib.pyplot as plt
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk import sent_tokenize, word_tokenize

lemmatizer = WordNetLemmatizer()


def get_processed_query(body, title=""):
    if body is None or body == '':
        return None
    body = title + ' ' + body
    desc = re.sub('<code>.*?</code>', '', body, flags=re.DOTALL)  # remove code from description
    desc = re.sub(r'<[^>]+>', '', desc, flags=re.DOTALL)  # remove all tags
    desc = re.sub(r'\s+', ' ', desc, flags=re.DOTALL)  # replace all new lines and multiple spaces with single space
    desc = re.sub(r'[^\x00-\x7f]', r'', desc)  # remove all non-ascii characters
    desc = re.sub('[' + string.punctuation + ']', '', desc)  # remove all punctuations
    desc = desc.lower()

    tokens = []

    raw_sentences = sent_tokenize(desc)
    for raw_sentence in raw_sentences:
        tokenized_text = word_tokenize(raw_sentence)
        tokens.extend(tokenized_text)

    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in en_stops]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens


tm_folder = glob.glob("../Data/TopicModelling/*")
print(tm_folder)
print("\n\n\n")

output_folder = '../Data/Processed/'
data = []
# i = 200
for file in tm_folder:

    print("Processing file: "+str(file))
    with open(file, encoding="utf8") as ff:
        for item in ff.read().strip().split('\n'):
            # if i<0:
            #     break
            post = json.loads(str(item))
            text_for_lda = get_processed_query(post['body'], post['title'])
            data.append({'tag': post['tags'],
                         'text_for_lda': text_for_lda
                         })
            # i-=1


print(data[:2])
# data = data[:500]
from gensim import corpora
dictionary = corpora.Dictionary([each_post['text_for_lda'] for each_post in data])

corpus = [dictionary.doc2bow(each_post['text_for_lda']) for each_post in data]


import pickle
pickle.dump(corpus, open('../Data/Processed/corpus.pkl', 'wb'))
dictionary.save('dictionary.gensim')

import gensim
NUM_TOPICS = 15
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=30, random_state=389)
ldamodel.save('model5.gensim')



topics = ldamodel.show_topics(num_topics = NUM_TOPICS, num_words=30, formatted=False)

# Printing topics
for topic in topics:
    print(topic)


from collections import Counter
topics = ldamodel.show_topics(num_topics = NUM_TOPICS, num_words=20, formatted=False)
data_flat = [word  for each_post in data for word in each_post['text_for_lda']]
print(len(data_flat))
counter = Counter(data_flat)
# ================================================================
# Generating data for Word Count and Importance of Topic Keywords
# ================================================================
out = []
for i, topic in topics:
    for word, weight in topic:
        out.append([word, i, weight, counter[word]])

print(len(out))
df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])
df = df[df['topic_id'].isin([0,2,3,4,5,6,7,8,10,11,12,14])]
df.to_csv('../Data/ChartData/GridData.csv', index=False)

#
# 0 tableau
# 1
# 2 elixir
# 3 matplotlib
# 4 selenium
# 5 joomla
# 6 talend
# 7 neo4j
# 8 apache/ tomcat
# 9
# 10 ubuntu
# 11 apache
# 12 database
# 13
# 14 keras


# ===========================================================================
# Generating data for "Distribution of Document Word Counts by Document Topic
# ===========================================================================
def format_topics_sentences(ldamodel=None, corpus=corpus, texts=data):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list
        # print(row)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)

data_ready = [each_post['text_for_lda'] for each_post in data]
df_topic_sents_keywords = format_topics_sentences(ldamodel=ldamodel, corpus=corpus, texts=data_ready)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
df_dominant_topic.head(10)

pd.options.display.max_colwidth = 100

sent_topics_sorteddf_mallet = pd.DataFrame()
sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

for i, grp in sent_topics_outdf_grpd:
    sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet,
                                             grp.sort_values(['Perc_Contribution'], ascending=False).head(1)],
                                            axis=0)

# Reset Index
sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

# Format
sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Representative Text"]

# Show
sent_topics_sorteddf_mallet.head(10)

doc_lens = [len(d) for d in df_dominant_topic.Text]


import matplotlib.colors as mcolors
cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

fig, axes = plt.subplots(2, 2, figsize=(16, 14), dpi=160, sharex=True, sharey=True)


freq_chart = []
for i in range(NUM_TOPICS):
    if i in [0,2,3,4,5,6,7,8,10,11,12,14]:
        df_dominant_topic_sub = df_dominant_topic.loc[df_dominant_topic.Dominant_Topic == i, :]
        doc_lens = [len(d) for d in df_dominant_topic_sub.Text]
        for d in df_dominant_topic_sub.Text:
            freq_chart.append({"topic_id" : i, "word_count" :len(d) })

pd.DataFrame(freq_chart).to_csv('../Data/ChartData/word_Distribution.csv', index=False)


# ================================================
# Generating data for "t-SNE Clustering of posts"
# ================================================
# Get topic weights and dominant topics ------------
from sklearn.manifold import TSNE
from bokeh.plotting import figure, show


# Get topic weights
topic_weights = []
for i, row_list in enumerate(ldamodel[corpus]):
    nn = [0 for x in range(NUM_TOPICS)]
    for x in row_list:
        nn[x[0]] = x[1]
    topic_weights.append(nn)

print(topic_weights[:5])
# for x in list(ldamodel[corpus]):
#     topic_weights.append(x[0])

# Array of topic weights
arr = pd.DataFrame(topic_weights).fillna(0).values

# Keep the well separated points (optional)
arr = arr[np.amax(arr, axis=1) > 0.35]

# Dominant topic number in each doc
topic_num = np.argmax(arr, axis=1)

# tSNE Dimension Reduction
tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
tsne_lda = tsne_model.fit_transform(arr)

# Plot the Topic Clusters using Bokeh
# output_notebook()

topic_num_df = pd.DataFrame(data=topic_num, columns=["topic_id"])
axis_df = pd.DataFrame(data=tsne_lda, columns=["X_axis", "Y_axis"])

f_final = pd.concat([topic_num_df, axis_df], axis=1)
f_final = f_final[f_final['topic_id'].isin([0,2,3,4,5,6,7,8,10,11,12,14])]
f_final.to_csv('../Data/ChartData/mds.csv',index=False)

mycolors = np.array([color for name, color in mcolors.XKCD_COLORS.items()])
plot = figure(title="t-SNE Clustering of {} LDA Topics".format(NUM_TOPICS),
              plot_width=900, plot_height=700)
plot.scatter(x=tsne_lda[:,0], y=tsne_lda[:,1], color=mycolors[topic_num])
show(plot)
#
