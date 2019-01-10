from nltk.corpus import stopwords

# basic
EMB_DIM = 300
STOPS = set(stopwords.words('english'))
SENTI_STOPS = STOPS.copy()
SENTI_STOPS.difference_update({'no', 'not'})
SENTI_STOPS.update({'.'})

# dataset names
SIMILARITY = 'similarity'
ANALOGY = 'analogy'
BROWN_POS = 'brown-pos'
WSJ_POS = 'wsj-pos'
CHUNKING = 'chunking'
SENTIMENT = 'sentiment'
NEWS = 'news'

ALL_DS = [SIMILARITY, ANALOGY, BROWN_POS, WSJ_POS, CHUNKING, SENTIMENT, NEWS]
SUP_DS = [BROWN_POS, WSJ_POS, SENTIMENT, NEWS, CHUNKING]