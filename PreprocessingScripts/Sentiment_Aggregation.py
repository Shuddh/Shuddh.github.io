
import datetime
import glob
import json
import re
import nltk
from nltk.corpus import stopwords
en_stops = set(stopwords.words('english'))
nltk.download("wordnet")
nltk.download("stopwords")
nltk.download('punkt')

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn

lemmatizer = WordNetLemmatizer()
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sid_obj = SentimentIntensityAnalyzer()
from nltk import sent_tokenize, word_tokenize, pos_tag


def penn_to_wn(tag):
    """
    Convert between the PennTreebank tags to simple Wordnet tags
    """
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None


def parse_get_week(text):
    dt = text.split()
    try:
      cur_dt = datetime.datetime.strptime(dt[0], '%Y-%m-%d')
      # return cur_dt - datetime.timedelta(days=cur_dt.weekday()) # returns first monday of the week
      return cur_dt
    except ValueError:
      return None


def get_processed_query(body, title=""):
    postive_words = set()
    negative_words = set()
    if body is None or body == '':
        return None
    body = title + ' ' + body
    desc = re.sub('<code>.*?</code>', '', body, flags=re.DOTALL)  # remove code from description
    desc = re.sub(r'<[^>]+>', '', desc, flags=re.DOTALL)  # remove all tags
    desc = re.sub(r'\s+', ' ', desc, flags=re.DOTALL)  # replace all new lines and multiple spaces with single space
    desc = re.sub(r'[^\x00-\x7f]', r'', desc)  # remove all non-ascii characters
    # desc = re.sub('[' + string.punctuation + ']', '', desc)  # remove all punctuations
    desc = desc.lower()

    # desc = clean_text(desc)

    raw_sentences = sent_tokenize(desc)
    for raw_sentence in raw_sentences:
        tagged_sentence = pos_tag(word_tokenize(raw_sentence))

        for word, tag in tagged_sentence:
            wn_tag = penn_to_wn(tag)
            if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
                continue

            lemma = lemmatizer.lemmatize(word, pos=wn_tag)
            if not lemma:
                continue

            synsets = wn.synsets(lemma, pos=wn_tag)
            if not synsets:
                continue

            # Take the first sense, the most common
            synset = synsets[0]
            swn_synset = swn.senti_synset(synset.name())
            if swn_synset.pos_score()> 0:
                postive_words.add(word)
            if swn_synset.neg_score()> 0:
                negative_words.add(word)

    polarity_dict = sid_obj.polarity_scores(desc)

    if polarity_dict['compound'] >= 0.05:
        overall_sentiment= 'positive'
    elif (polarity_dict['compound'] > -0.05) and (polarity_dict['compound'] < 0.05):
        overall_sentiment = 'neutral'
    else:
        overall_sentiment = 'negative'
    return polarity_dict, list(postive_words), list(negative_words), overall_sentiment


questions_folder = glob.glob("../Data/Questions/2018/*")
questions_folder = questions_folder[9:]
print(questions_folder)
print("\n\n\n")
# questions_folder = ['../Data/Questions/2018\\Q_000000000003.json', '../Data/Questions/2018\\Q_000000000004.json', '../Data/Questions/2018\\Q_000000000005.json', '../Data/Questions/2018\\Q_000000000006.json', '../Data/Questions/2018\\Q_000000000007.json', '../Data/Questions/2018\\Q_000000000008.json', '../Data/Questions/2018\\Q_000000000009.json']
output_folder = '../Data/Processed/'
i = 9
for file in questions_folder:
    data = []
    # print(file, len(data))
    print("Processing file: "+str(file))
    with open(file, encoding="utf8") as ff:
        for item in ff.read().strip().split('\n'):
            post = json.loads(str(item))
            if 'java' in post['tags'] and 'javascript' not in post['tags']:
                polarity_dict, positive_words, negative_words, overall_sentiment = get_processed_query(post['body'], post['title'])
                data.append({'tags': post['tags'].split('|'),
                             'creation_date': str(parse_get_week(post['creation_date'])),
                             'negative_score': polarity_dict['neg'],
                             'positive_score': polarity_dict['pos'],
                             'compound_score': polarity_dict['compound'],
                             'neutral_score': polarity_dict['neu'],
                             'positive_words': positive_words,
                             'negative_words': negative_words,
                             'pos_post': 1 if overall_sentiment == 'positive' else 0,
                             'neg_post': 1 if overall_sentiment == 'negative' else 0,
                             'neu_post': 1 if overall_sentiment == 'neutral' else 0
                             })
    with open('../Data/Processed/Q'+str(i)+'.json', 'w') as fout:
        json.dump(data, fout)
    i+=1


