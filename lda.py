from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import string
import nltk, re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
import sys
reload(sys)
sys.setdefaultencoding('utf8')

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
#stemmer = SnowballStemmer("english")

def tokenize_and_stem(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [t for t in filtered_tokens]
    return stems

def clean(doc):
    doc = doc.decode('ascii',errors='ignore').strip()
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized
'''
f = open('msft/content.txt','r')
doc_complete = f.readlines()
doc_complete = doc_complete[:1000]
doc_clean = [clean(doc) for doc in doc_complete]

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.0002,stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,4))
tfidf_matrix = tfidf_vectorizer.fit_transform(doc_clean)
print(tfidf_matrix.shape)
'''


# def fil(word):
#     word = ''.join(ch for ch in word if ch.isalnum() or ch is ' ')
#     return word
#
# words = [ fil(word) for word in file("content.txt", "r").read().split()]
# counts = Counter(words)

# Importing Gensim
#import gensim
#from gensim import corpora

# Creating the term dictionary of our courpus, where every unique term is assigned an index.
#dictionary = corpora.Dictionary(doc_clean)
# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
#doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
#Lda = gensim.models.ldamodel.LdaModel
#Running and Trainign LDA model on the document term matrix.
#ldamodel = Lda(doc_term_matrix, num_topics=1000, id2word = dictionary, passes=50)


#print(ldamodel.print_topics(num_topics=1000, num_words=100))

# X = lda.datasets.load_reuters()
# vocab = lda.datasets.load_reuters_vocab()
# X.shape
# X.sum()
# model = lda.LDA(n_topics=20, n_iter=1500, random_state=1)
# model.fit(X)  # model.fit_transform(X) is also available
# topic_word = model.topic_word_  # model.components_ also works
# n_top_words = 8
# for i, topic_dist in enumerate(topic_word):
# 	topic_words = np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
# 	print('Topic {}: {}'.format(i, ' '.join(topic_words)))
