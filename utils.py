import plotly
from plotly.offline import plot
plotly.__version__
import plotly.figure_factory as ff
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import operator
from nltk.corpus import stopwords
from nltk.util import ngrams

def read_label(label_filename):
    labelfile = open(label_filename,'r')
    #labelfile = open('msft/rlabel.txt', 'r')
    #labelfile = open('msft/clabel.txt', 'r')
    #labelfile = open('testLabels.txt','r')
    #labelfile = open('./interestlabels.txt','r')
    labelfile = [x.strip() for x in labelfile]
    rev_label = {}
    ind = 0
    for label in labelfile:
        rev_label[label.strip()] = ind
        ind+=1
    print('labels read\n')
    return labelfile,rev_label

def read_sequences(sequence_filename):
    sequence_file = open(sequence_filename, 'r')
    #sequence_file = open('rsequences.txt', 'r')
    #sequence_file = open('csequences.txt', 'r')
    #sequence_file = open('testSequences.txt','r')
    #sequence_file = open('./profileinterest20.txt','r')
    sequence_file = sequence_file.readlines()
    print('sequences read\n')
    return sequence_file

def read_clusters(minClusterSize):
    clusters = {}
    sizewise_dic = {}
    #f = open('clusters.txt', 'r')
    #f = open('cSentNew60k-2kclusters.txt','r')
    f = open('cSent-clusters100.txt')
    f = f.readlines()
    for line in f:
        line = line.strip()
        arr = line.split('\t')
        if(len(arr)!=2):
            continue
        clust_id = arr[0].split(':')[1]
        clus_mem = [data.strip() for data in arr[1][1:-1].split('\', \'')]
        clusters[clust_id] = clus_mem
        if len(clus_mem) not in sizewise_dic:
            sizewise_dic[len(clus_mem)] = []
        sizewise_dic[len(clus_mem)].append(clust_id)

    print('Size vs num of clusters:\n')
    valid_clusterids = []
    for key, val in sizewise_dic.iteritems():
        print(str(key) + '\t' + str(len(val)) + '\n')
        if (key > minClusterSize):
            valid_clusterids.extend(val)

    return clusters, valid_clusterids


def read_matrix(filename):
    f = open(filename,'r')
    f = f.readlines()
    red_labels = f[0].strip().split('\t')
    red_X = np.zeros((len(red_labels),len(red_labels)))
    f = f[1:]
    row = 0
    for line in f:
        line = line.strip()
        if(len(line)>0):
            red_X[row] = line.strip().split()
            row+=1
    return red_X,red_labels

def findMax(memA, memB, rev_label, X):
    maxNum = 0
    for A in memA:
        for B in memB:
            rAt = min(rev_label[A], rev_label[B])
            rBt = max(rev_label[A], rev_label[B])
            if X[rAt][rBt] > maxNum:
                maxNum = X[rAt][rBt]
    return maxNum

def combineClusters(clusters,rev_label, X):
    print('Generating reduced matrix...\n')
    # Generate reduced matrix
    reducedX = np.zeros((len(clusters), len(clusters)))
    new_index_dic = {}  # store clusterId vs reducedX index
    nameId = 0
    for key1 in clusters:
        for key2 in clusters:
            if key1 not in new_index_dic:
                new_index_dic[key1] = nameId
                nameId += 1
            if key2 not in new_index_dic:
                new_index_dic[key2] = nameId
                nameId += 1
            if new_index_dic[key1] < new_index_dic[key2]:
                linkage_metric = findMax(clusters[key1], clusters[key2], rev_label, X)
                reducedX[new_index_dic[key1]][new_index_dic[key2]] = linkage_metric
                #reducedX[new_index_dic[key2]][new_index_dic[key1]] = linkage_metric
            else:
                reducedX[new_index_dic[key1]][new_index_dic[key2]] = 0
    return reducedX,new_index_dic

def combineUIClusters(valid_clusters,rev_label, X):
    print('Generating reduced matrix...\n')
    # Generate reduced matrix
    reducedX = np.zeros((len(valid_clusters), len(valid_clusters)))
    new_index_dic = {}  # store clusterId vs reducedX index
    nameId = 0
    for key1 in valid_clusters:
        for key2 in valid_clusters:
            if key1 not in new_index_dic:
                new_index_dic[key1] = nameId
                nameId += 1
            if key2 not in new_index_dic:
                new_index_dic[key2] = nameId
                nameId += 1
            if new_index_dic[key1] != new_index_dic[key2]:
                linkage_metric = X[rev_label[key1]][rev_label[key2]]
                reducedX[new_index_dic[key1]][new_index_dic[key2]] = linkage_metric
                reducedX[new_index_dic[key2]][new_index_dic[key1]] = linkage_metric
            else:
                reducedX[new_index_dic[key1]][new_index_dic[key2]] = 0
    return reducedX,new_index_dic

def generate_dendrogram(reducedX,new_labels, dendroname):
    fig = ff.create_dendrogram(reducedX, orientation='left', labels=new_labels)
    print("dendogram created\n")
    fig['layout'].update({'width': 1000, 'height': 800})
    print("plotting started\n")
    plot(fig, filename=dendroname)
    print("plotting done\n")

def generate_clouds(tf,minClusSize):
    plt.figure()
    figureId = 1
    gridSize = 4
    count = 0
    for key in tf:
        if (len(tf[key]) <= minClusSize):
            continue
        ax = plt.subplot(gridSize, gridSize, figureId)
        ax.set_title('Cluster :'+str(key))
        figureId += 1
        wordcloud = WordCloud(background_color="white", max_font_size=40).generate_from_frequencies(tf[key])
        plt.imshow(wordcloud, interpolation="bilinear")
        count += 1
        if figureId > gridSize * gridSize:
            plt.axis("off")
            plt.figure()
            figureId = 1
    plt.axis("off")
    plt.show()

def process_api_response(apiresponse):
    apiresponse = apiresponse.strip()
    apiresponse = apiresponse[1:-1]
    response_arr = apiresponse.split(',')
    concepts = []
    for entry in response_arr:
        concepts.append(entry.split(':')[0][1:-1])
    return set(concepts)

def read_knowledge_base():
    f = open('/mnt1/resources/insights/apiScore/combinedApi.txt', 'r')
    f = f.readlines()
    data_dic = {}

    for line in f:
        arr = line.split('\t')
        if(len(arr)==2):
            if arr[0] not in data_dic:
                data_dic[arr[0]] = set()
            data_dic[arr[0]].update(process_api_response(arr[1]))

    return data_dic


def get_cluster_label(cluster, dic, knowledge_threshold, debug_file):
    candidate_label = {}
    most_freq_word = {}
    for value in cluster:
        if value in dic:
            for concept in dic[value]:
                concept = concept.strip()
                if concept not in candidate_label:
                    candidate_label[concept]=0
                candidate_label[concept]+=1
        else:
            for word in value.split():
                word = word.strip()
                if word not in most_freq_word:
                    most_freq_word[word]=0
                most_freq_word[word]+=1
    backup = ''
    if len(candidate_label)>0:
        candidate_label_sets = sorted(candidate_label.items(),key=operator.itemgetter(1))
        if(candidate_label_sets[len(candidate_label_sets)-1][1]>knowledge_threshold):
            debug_file.write('Mode 1\n')
            debug_file.write(str(cluster)+'\n')
            debug_file.write(str(candidate_label_sets)+'\n')
            debug_file.write(candidate_label_sets[len(candidate_label_sets)-1][0]+'\n\n')
            return candidate_label_sets[len(candidate_label_sets)-1][0]
        else:
            backup = candidate_label_sets[len(candidate_label_sets)-1][0]

    most_freq_word_sets = sorted(most_freq_word.items(), key=operator.itemgetter(1))
    if len(most_freq_word_sets)>0:
        debug_file.write('Mode 2\n')
        debug_file.write(str(cluster) + '\n')
        debug_file.write(str(most_freq_word_sets) + '\n')
        debug_file.write(most_freq_word_sets[len(most_freq_word_sets)-1][0] + '\n\n')
        return most_freq_word_sets[len(most_freq_word_sets)-1][0]
    if len(backup)>0:
        return backup
    else:
        return 'NOLABEL'


def filter(word):
    word = ''.join(ch for ch in word if ch.isalnum() or ch is ' ')
    return word

def get_cluster_label_sentences(cluster, dic, knowledge_threshold, debug_file):
    candidate_label = {}
    most_freq_word = {}
    filtered_words = open('msft/filtered_tf.txt','r')
    filtered_words = [word.strip() for word in filtered_words.readlines()]
    filtered_words = set(filtered_words)
    if len(cluster) <= 1:
        return 'NOLABEL'
    for value in cluster:
        value = filter(value)
        unigrams = value.split()
        bigrams = ngrams(unigrams, 2)
        trigrams = ngrams(unigrams, 3)
        combined = unigrams.extend(bigrams).extend(trigrams)
        for word in combined:
            if word in filtered_words:
                for concept in dic[word]:
                    concept = concept.strip()
                    if concept not in candidate_label:
                        candidate_label[concept] = 0
                    candidate_label[concept] += 1
            else:
                # for word in value.split():
                word = word.strip()
                if word not in most_freq_word:
                    most_freq_word[word] = 0
                most_freq_word[word] += 1
    backup = ''
    if len(candidate_label) > 0:
        candidate_label_sets = sorted(candidate_label.items(), key=operator.itemgetter(1))
        if (candidate_label_sets[len(candidate_label_sets) - 1][1] > knowledge_threshold):
            debug_file.write('Mode 1\n')
            debug_file.write(str(cluster) + '\n')
            debug_file.write(str(candidate_label_sets) + '\n')
            debug_file.write(candidate_label_sets[len(candidate_label_sets) - 1][0] + '\n\n')
            return candidate_label_sets[len(candidate_label_sets) - 1][0]
        else:
            backup = candidate_label_sets[len(candidate_label_sets) - 1][0]
    most_freq_word_sets = sorted(most_freq_word.items(), key=operator.itemgetter(1))
    if len(most_freq_word_sets) > 0:
        debug_file.write('Mode 2\n')
        debug_file.write(str(cluster) + '\n')
        debug_file.write(str(most_freq_word_sets) + '\n')
        debug_file.write(most_freq_word_sets[len(most_freq_word_sets) - 1][0] + '\n\n')
        filtered_words = [word for word in most_freq_word_sets if
                          word[0].lower() not in stopwords.words('english') and len(filter(word[0])) > 3]
        if len(filtered_words) > 0:
            return filtered_words[len(filtered_words) - 1][0]
        return 'NOLABEL'
    if len(backup) > 0:
        return backup
    else:
        return 'NOLABEL'

# def get_cluster_label_sentences(cluster, dic, knowledge_threshold, debug_file):
#     candidate_label = {}
#     most_freq_word = {}
#     if len(cluster)<=1:
#         return 'NOLABEL'
#     for value in cluster:
#         for word in value.split():
#             if word in dic:
#                 for concept in dic[word]:
#                     concept = concept.strip()
#                     if concept not in candidate_label:
#                         candidate_label[concept]=0
#                     candidate_label[concept]+=1
#             else:
#                 #for word in value.split():
#                 word = word.strip()
#                 if word not in most_freq_word:
#                     most_freq_word[word]=0
#                 most_freq_word[word]+=1
#     backup = ''
#     if 0 and len(candidate_label)>0:
#         candidate_label_sets = sorted(candidate_label.items(),key=operator.itemgetter(1))
#         if(candidate_label_sets[len(candidate_label_sets)-1][1]>knowledge_threshold):
#             debug_file.write('Mode 1\n')
#             debug_file.write(str(cluster)+'\n')
#             debug_file.write(str(candidate_label_sets)+'\n')
#             debug_file.write(candidate_label_sets[len(candidate_label_sets)-1][0]+'\n\n')
#             return candidate_label_sets[len(candidate_label_sets)-1][0]
#         else:
#             backup = candidate_label_sets[len(candidate_label_sets)-1][0]
#     most_freq_word_sets = sorted(most_freq_word.items(), key=operator.itemgetter(1))
#     if len(most_freq_word_sets)>0:
#         debug_file.write('Mode 2\n')
#         debug_file.write(str(cluster) + '\n')
#         debug_file.write(str(most_freq_word_sets) + '\n')
#         debug_file.write(most_freq_word_sets[len(most_freq_word_sets)-1][0] + '\n\n')
#         filtered_words = [word for word in most_freq_word_sets if word[0].lower() not in stopwords.words('english') and len(filter(word[0]))>3]
#         if len(filtered_words) > 0:
#             return filtered_words[len(filtered_words)-1][0]
#         return 'NOLABEL'
#     if len(backup)>0:
#         return backup
#     else:
#         return 'NOLABEL'