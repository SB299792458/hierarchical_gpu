import constants, utils, operator, os
import generateData, generatePlotReducedDendro


def read_labels_detected():
    labels_detected = open('labelsDetected.txt', 'r')
    data_dic = {}
    id = 0
    for line in labels_detected:
        arr = line.strip().split('\t')
        lines = arr[0].split(constants.value_separator)
        labels = arr[1].split(constants.label_separator)
        data_dic[id] = {'lines': lines, 'labels': labels}
        id += 1
    return data_dic


def generate_dendrogram():
    data_dic = read_labels_detected()
    labels_op = open('labels_op.txt', 'w')

    for key, val in data_dic.iteritems():
        labels_op.write(','.join(val['labels']))
        labels_op.write('\n')

    labels_op.close()
    output_file_prefix = 'levels'
    generateData.generate_data('labels_op.txt', constants.label_separator, 'levels')
    # Done separately
    # os.system(constants.hier_code_root + "/hier " + output_file_prefix + 'wvec.dat dummy.txt 1 > '+ output_file_prefix+'logs.txt')

    clusters, rev_label, X = generatePlotReducedDendro.generate(output_file_prefix)
    labels , rev_label = utils.read_label(output_file_prefix + 'label.txt')
    new_labels = [ str(','.join(label.strip().split(',')[:3])) for label in labels]
    utils.generate_dendrogram(X, new_labels, 'dendrogram_with_labels')

    print("plotting done\n")
