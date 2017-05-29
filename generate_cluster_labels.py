import constants, filenames
import operator
import sys

import numpy as np
from nltk.corpus import stopwords
from nltk.util import ngrams

import lda

reload(sys)

sys.setdefaultencoding("utf-8")

def process_api_response(apiresponse):
    apiresponse = apiresponse.strip()
    apiresponse = apiresponse[1:-1]
    response_arr = apiresponse.split(',')
    concepts = []
    for entry in response_arr:
        concepts.append(entry.split(':')[0][1:-1])
    return set(concepts)

def read_knowledge_base():
    f = open('/mnt1/resources/insights/apiScore/combinedApiSmall.txt', 'r')
    f = f.readlines()
    data_dic = {}
    for line in f:
        arr = line.split('\t')
        if(len(arr)==2):
            if arr[0] not in data_dic:
                data_dic[arr[0]] = set()
            data_dic[arr[0]].update(process_api_response(arr[1]))
    return data_dic

def read_clusters(minClusterSize):
    clusters = {}
    sizewise_dic = {}
    #f = open('clusters.txt', 'r')
    # f = open('cSent-clusters100.txt')
    f = open('cSentFilter60k-clusters.txt','r')
    #f = f.readlines()
    #f = ['clusterId:172    @Xbox Thanks for the chance!_#_Thanks @Xbox  https://t.co/u20gA2EKtf_#_@Xbox  Thanks for chance!_#_@Xbox Ikr! Lol thanks_#_@Xbox thank you_#_@Xbox Oh thank you! _#_@Xbox thanks for the chance_#_@Xbox @XboxSupport thanks!!!!_#_@Xbox haha well thanks for the reference_#_@Xbox thanks for sharing_#_thank you @Xbox"_#_@Xbox thank you for the chance Xbox _#_@Xbox thank u guys  https://t.co/DkPjZPzj2b_#_@XboxSupport thanks Xbox! _#_@XboxSupport @Xbox thanks_#_@XboxSupport thanks! XBOX ONE']
    for line in f:
        line = line.strip()
        arr = line.split('\t')
        if(len(arr)!=2):
            continue
        clust_id = arr[0].split(':')[1]
        clus_mem = [data.strip() for data in arr[1].split(constants.value_separator)]
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

def filter(word):
    word = ''.join(ch for ch in word if ch.isalnum() or ch is ' ')
    return word

def combine(string1, string2):
    words1 = string1.split()
    words2 = string2.split()
    m = len(words1)
    n = len(words2)
    if words1[0]==words2[n-1]:
        temp = words1
        words1 = words2
        words2 = temp
    m = len(words1)
    n = len(words2)
    i=m-1
    j=0
    common = []
    while(i>=0 and j<=n-1 and words1[i]==words2[j]):
        common.append(words1[i])
        i-=1
        j+=1
    words1 = words1[:i + 1]
    words1.extend(common)
    words1.extend(words2[j:])
    return ' '.join(words1)

def overlap(string1,string2):
    words1 = string1.split()
    words2 = string2.split()
    if words1[0] == words2[len(words2)-1] or words2[0] == words1[len(words1)-1]:
        return True
    return False

def combine2formlabel(filtered_words, cluster_size):
    curr_label = filtered_words[0][0]
    len_curr_label = len(curr_label.split())
    labels_list = []
    freq = []

    for label_entry in filtered_words:
        freq.append(label_entry[1])
    p = max(2,max(np.percentile(np.array(freq), 90),cluster_size/2))
    #increase length of most frequent label
    for label_entry in filtered_words[1:]:
        if label_entry[1] < p:
            break
        label = label_entry[0]
        #combine if it can be combined
        if len(label.split())>len_curr_label and overlap(curr_label,label):
            labels_list.append(combine(curr_label,label))
        #otherwise, add as it is, if there is no overlap
        elif not overlap(curr_label,label):
            labels_list.append(label)
    if len(labels_list)>0:
        return ','.join(labels_list)
    return curr_label

def get_cluster_label_sentences(cluster, dic, knowledge_threshold, debug_file, filtered_words):
    candidate_label = {}
    most_freq_word = {}
    if len(cluster) <= 1:
        return 'NOLABEL'
    cluster_size = len(cluster)
    for value in cluster:
        value = lda.clean(value)#filter(value)
        #value = value.encode('ascii',errors='ignore')
        combined = lda.tokenize_and_stem(value)#value.split()
        bigrams = ngrams(combined, 2)
        trigrams = ngrams(combined, 3)
        bigrams_list = list(bigrams)
        trigrams_list = list(trigrams)
        bigrams_list.extend(trigrams_list)
        for value in bigrams_list:
            combined.append(' '.join(value))
        for word in combined:
            if word in filtered_words and word in dic:
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
        candidate_label_sets = sorted(candidate_label.items(), key=operator.itemgetter(1), reverse=True)
        if (candidate_label_sets[len(candidate_label_sets) - 1][1] > knowledge_threshold):
            debug_file.write('Mode 1\n')
            debug_file.write(str(cluster) + '\n')
            debug_file.write(str(candidate_label_sets) + '\n')
            debug_file.write(candidate_label_sets[len(candidate_label_sets) - 1][0] + '\n\n')
            return candidate_label_sets[0][0]
        else:
            backup = candidate_label_sets[0][0]
    most_freq_word_sets = sorted(most_freq_word.items(), key=operator.itemgetter(1), reverse=True)
    if len(most_freq_word_sets) > 0:
        debug_file.write('Mode 2\n')
        debug_file.write(str(cluster) + '\n')
        debug_file.write(str(most_freq_word_sets) + '\n')
        debug_file.write(most_freq_word_sets[len(most_freq_word_sets) - 1][0] + '\n\n')
        #optimize
        fil_words = [word for word in most_freq_word_sets if
                          word[0].lower() not in stopwords.words('english') and len(filter(word[0])) > 3]
        if len(fil_words) > 0:
            return combine2formlabel(fil_words, cluster_size)
        return 'NOLABEL'

    if len(backup) > 0:
        return backup
    else:
        return 'NOLABEL'
def generate_labels():
    clusters, valid_clusterids = read_clusters(0)
    data_dic = read_knowledge_base()
    op_file = open(filenames.cluster_labels_file,'w')
    debug_file = open('debug_file.txt','w')
    #filtered_words = open('msft/filtered_tf.txt', 'r')
    filtered_words = open('tf-idf.txt', 'r')
    filtered_words = [unicode(word.strip()) for word in filtered_words.readlines()]
    filtered_words = set(filtered_words)
    for id,cluster in clusters.iteritems():
        label = get_cluster_label_sentences(cluster, data_dic, 2, debug_file,filtered_words)
        op_file.write(str(constants.value_separator.join(cluster)))
        op_file.write('\t'+label+'\n\n')
    op_file.close()
    debug_file.close()