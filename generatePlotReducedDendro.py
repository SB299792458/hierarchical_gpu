import plotly
plotly.__version__
import numpy as np
import operator
import utils
import constants


def generate(output_file_prefix):
    labelfile, rev_label = utils.read_label(output_file_prefix + 'label.txt')
    sequence_file = utils.read_sequences(output_file_prefix + 'sequences.txt')
    cluster_file = open(output_file_prefix+'clusters.txt', 'w')
    # imp
    N = len(sequence_file) + 1
    threshold = 10

    print('N = ' + str(N) + '\n')
    height_dic = {}
    X = np.zeros((N, N))
    print("files loaded\n")
    members = {i: [i] for i in range(0, N)}
    print("members initialized\n")
    clId = 0
    clustID = {labelfile[i]: i for i in range(0, N)}
    for line in sequence_file:
        line = line.strip()
        seq = line.split('\t')
        seqA = int(seq[0])
        seqB = int(seq[1])
        seqC = int(seq[2])
        if seqC not in members:
            members[seqC] = []
        if (seqA < N):
            height_dic[seqA] = 0
        if (seqB < N):
            height_dic[seqB] = 0

        members[seqC].extend(members[seqA])
        members[seqC].extend(members[seqB])
        height_dic[seqC] = max(height_dic[seqA], height_dic[seqB]) + 1
        for i in members[seqA]:
            for j in members[seqB]:
                X[i][j] = height_dic[seqC]
                X[j][i] = height_dic[seqC]
                if X[i][j] <= threshold:
                    if labelfile[i] not in clustID:
                        clId += 1
                        clustID[labelfile[i]] = clId
                    clustID[labelfile[j]] = clustID[labelfile[i]]

    print("matrix created\n")
    clusters = {}
    for key, value in clustID.iteritems():
        if value not in clusters:
            clusters[value] = list()
        clusters[value].append(key)

    for key, value in clusters.iteritems():
        cluster_file.write('clusterId:' + str(key) + '\t')
        cluster_file.write(str(constants.value_separator.join(value)))
        cluster_file.write('\n')
    cluster_file.close()
    print("clusters written\n")

    return clusters, rev_label, X