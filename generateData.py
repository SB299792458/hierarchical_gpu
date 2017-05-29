import collections, time, numpy
from nltk.corpus import stopwords
import sys, logging
from multiprocessing.pool import ThreadPool
from functools import partial
import filenames
reload(sys)
sys.setdefaultencoding("utf-8")


def transpose(inputfile, outputfile, labelfile):
    f = inputfile.readlines()
    f = f[1:]
    X = []
    skipped = 0
    for ind, line in enumerate(f):
        line = line.strip()
        arr = line.split('\t')
        if len(arr) != 2:
            print('Len not 2 error:\n')
            print(line)
            skipped += 1
            continue
        labelfile.write(arr[0] + '\n')
        arr = arr[1].strip().split(',')
        X.append(arr)
    X = map(list, zip(*X))
    print len(X)
    print len(X[0])
    single_d = [X[i][j] for i in range(0, len(X)) for j in range(0, len(X[i]))]
    outputfile.write(str(len(X[0])) + '\n')
    outputfile.write(str(10) + '\n')
    outputfile.write(str(len(X)) + '\n')
    for i in single_d:
        outputfile.write(i + '\n')
    outputfile.close()
    labelfile.close()
    print('Skipped :' + str(skipped) + '\n')


def load_words(tfidffile):
    tf_idf_file = open(tfidffile, 'r')
    tf_idf_words = set()
    for line in tf_idf_file:
        tf_idf_words.add(line.strip())
    return tf_idf_words


def load_word2vec(word2vecfile):
    word2vec = open(word2vecfile, 'r')
    wordvec_dict = {}
    for line in word2vec:
        ind = line.index(' ')
        wordvec_dict[line[0:ind].strip()] = line[ind + 1:].strip()
    return wordvec_dict


def addWvec(separator, word2vec_dic, tf_idf_words, line):
    line = line.encode('ascii', errors='ignore')
    line = line.strip()
    line = ''.join(ch for ch in line if ch.isalnum() or ch is ' ')
    line = line.lower()
    if(len(separator)>0):
        line = line.replace(separator,' ')
    wordslist = line.split()  # vector_filtered
    N = len(wordslist)
    if N < 1:
        return []
    vec = numpy.zeros((300))
    if wordslist[0] in word2vec_dic and wordslist[0] in tf_idf_words:
        vec = [float(v.strip()) for v in word2vec_dic[wordslist[0]].split()]
    if N < 2:
        return vec
    for word in wordslist[1:]:
        if word in word2vec_dic and word in tf_idf_words:
            vec2 = [float(v.strip()) for v in word2vec_dic[word].split()]
            for i in range(0, len(vec)):
                vec[i] += vec2[i]
    for i in range(0, len(vec)):
        vec[i] = vec[i] / N
    return vec


def generate_sentencevec(lines, separator, word2vec_dic, tf_idf_words):
    content_keys = {}
    content_keys_invalid = {}
    p = ThreadPool(1)
    addwvec_func = partial(addWvec, separator, word2vec_dic, tf_idf_words)
    results = p.map(addwvec_func, lines)
    for idx, line in enumerate(lines):
        svec = results[idx]
        if (len(svec) != 300):
            content_keys_invalid[line] = svec
            continue
        content_keys[line] = svec
    return content_keys, content_keys_invalid


# generate word2vec representation of data
def generate_data(input_file, separator, output_file_prefix):
    word2vec_dic = load_word2vec(word2vecfile=filenames.word2vecfile)
    tf_idf_words = load_words(tfidffile=filenames.tfidffile)
    print('loaded word vec\n')
    content = open(input_file, 'r').readlines()
    content[:10]
    content_keys, content_keys_invalid = generate_sentencevec(content, separator, word2vec_dic, tf_idf_words)
    print('loaded lines\n')
    content_data = open(output_file_prefix+'WvecFiltered.dat', 'w')
    content_data.write(str(len(content_keys)) + '\n')
    for key in content_keys:
        content_data.write(key.strip() + '\t' + str(list(content_keys[key]))[1:-1] + '\n')
    content_data.close()
    print('vectors extracted\n')
    transpose(open(output_file_prefix+'WvecFiltered.dat', 'r'), open(output_file_prefix+'wvec.dat', 'w'), open(output_file_prefix+'label.txt', 'w'))
    print('transpose done\n')
    print('Done generation')
