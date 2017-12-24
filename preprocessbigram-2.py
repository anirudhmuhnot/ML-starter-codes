import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk import sent_tokenize, word_tokenize, pos_tag
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from nltk.corpus import stopwords
import re
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score
from unidecode import unidecode 
from nltk import word_tokenize
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import extract_bigram_feats,mark_negation
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
import re
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize
from nltk.sentiment.util import mark_negation
from sklearn.feature_extraction.text import TfidfVectorizer



train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
# test['Is_Response'] = np.nan
# df = pd.concat([train, test]).reset_index(drop=True)

# df['Browser_Used']= df['Browser_Used'].replace('Mozilla Firefox','Firefox')
# df['Browser_Used']= df['Browser_Used'].replace('Mozilla','Firefox')
# df['Browser_Used']= df['Browser_Used'].replace('Google Chrome','Chrome')
# df['Browser_Used']= df['Browser_Used'].replace('InternetExplorer','IE')
# df['Browser_Used']= df['Browser_Used'].replace('Internet Explorer','IE')
# df['Is_Response'] = df['Is_Response'].replace('happy',1)
# df['Is_Response'] = df['Is_Response'].replace('not happy',0)
 
# y = df['Is_Response']
# dummies_browser = pd.get_dummies(df['Browser_Used'],prefix="browser")
# dummies_devid = pd.get_dummies(df['Device_Used'],prefix="device")
# dummies_browser = dummies_browser.iloc[:,1:]
# dummies_devid = dummies_devid.iloc[:,1:]
# df = pd.concat([df,dummies_browser,dummies_devid],axis=1)
# df.drop(['Browser_Used','Is_Response','Device_Used'],axis=1,inplace=True)
# stops = set(stopwords.words("english"))
# def cleanData(text, lowercase = False, remove_stops = False, stemming = False):
#     txt = str(text)
#     txt = re.sub(r'[^A-Za-z0-9\s.]',r'',txt)
#     txt = re.sub(r'\n',r' ',txt)
    
#     if lowercase:
#         txt = " ".join([w.lower() for w in txt.split()])
        
#     if remove_stops:
#         txt = " ".join([w for w in txt.split() if w not in stops])
    
#     if stemming:
#         st = WordNetLemmatizer()
#         txt = " ".join([st.lemmatize(w,'v') for w in txt.split()])

#     return txt

# df['Description'] = df['Description'].map(lambda x: cleanData(x, lowercase=True, remove_stops=True, stemming=True))
# df.to_csv('finaldfs/ohedf.csv',index=False)
df = pd.read_csv('finaldfs/ohedf.csv')


# lemmatizer = WordNetLemmatizer()



# def penn_to_wn(tag):
#     """
#     Convert between the PennTreebank tags to simple Wordnet tags
#     """
#     if tag.startswith('J'):
#         return wn.ADJ
#     elif tag.startswith('N'):
#         return wn.NOUN
#     elif tag.startswith('R'):
#         return wn.ADV
#     elif tag.startswith('V'):
#         return wn.VERB
#     return None
 
# i = 0
# def swn_polarity(text):
#     """
#     Return a sentiment polarity: 0 = negative, 1 = positive
#     """
#     sentiment = 0.0
#     tokens_count = 0
  
 
#     raw_sentences = sent_tokenize(text)
#     for raw_sentence in raw_sentences:
#         tagged_sentence = pos_tag(word_tokenize(raw_sentence))
 
#         for word, tag in tagged_sentence:
#             wn_tag = penn_to_wn(tag)
#             if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
#                 continue
 
#             lemma = lemmatizer.lemmatize(word, pos=wn_tag)
#             if not lemma:
#                 continue
 
#             synsets = wn.synsets(lemma, pos=wn_tag)
#             if not synsets:
#                 continue
 
#             # Take the first sense, the most common
#             synset = synsets[0]
#             # sysnet1 = synsets[1]
#             swn_synset = swn.senti_synset(synset.name())
#             # swn_synset1 = swn.senti_synset(synset1.name())
#             sentiment += swn_synset.pos_score() - swn_synset.neg_score()
#             # sentiment += swn_synset1.pos_score() - swn_synset1.neg_score()
#             tokens_count += 1
 
#     # judgment call ? Default to positive or negative
#     if not tokens_count:
#         return -1
 
#     # sum greater than 0 => positive sentiment
#     if sentiment >= 0:
#         return 1
 
#     # negative sentiment
#     return 0


# print("SWN POL")
# df['swn_polarity']= df['Description'].map(lambda x: swn_polarity(x))
# df['swn_polarity'].to_csv('finaldfs/swn_polarity.csv',index=False)

# print("Vader Pol")
# vader = SentimentIntensityAnalyzer()
# def vader_polarity(text):
#     """Transform the output to a binary 0/1 result """
#     score = vader.polarity_scores(text)
#     return 1 if score['pos'] > score['neg'] else 0



# df['vader_pol'] = df['Description'].map(lambda x: vader_polarity(x))
# df['vader_pol'].to_csv('finaldfs/vader_pol.csv',index=False)

# df['vader_pol'] = pd.read_csv('vader_pol.csv')
swnpol = pd.read_csv('finaldfs/swn_polarity.csv',header=None)
print(df.tail())
swnpol.columns = ['swn_polarity']

print("Ngram 3")
tfidfvec = TfidfVectorizer(analyzer='word', ngram_range = (1,2),tokenizer=lambda text: mark_negation(word_tokenize(text)),max_features=9000)
# df.memory_usage(index=True).sum()


tfidfdata = tfidfvec.fit_transform(df['Description'])



tfidfdata = pd.DataFrame(tfidfdata.todense())
print(tfidfdata.tail())


# tfidfdata.to_csv('tfidf.csv',index=False)


print("Concat")
x = pd.concat([df,tfidfdata,swnpol],axis=1)

# df.to_csv('df.csv',index=False)
# ncount = pd.read_csv('negativecount.csv')

pcount = pd.read_csv('positivecount.csv')

print("Splitting")
# x = pd.concat([x,ncount,pcount],axis=1)
trainnew = x[:len(train)]
testnew = x[len(train):]
print("saving")
store = pd.HDFStore('data1.h5')
store['train']= trainnew
store['test'] = testnew



