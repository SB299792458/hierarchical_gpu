from neo4j.v1 import GraphDatabase, basic_auth

import constants


def gen_graph(data_dic):
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=basic_auth("neo4j", "qwerty"))
    session = driver.session()
    for key,val in data_dic.iteritems():
        session.run("CREATE (a:Content { id: {id}, lines: {lines}, labels: {labels}})", {"id":key ,"lines": val['lines'], "labels": val['labels']})

def read_labels_detected():
    labels_detected = open('labelsDetected.txt','r')
    data_dic = {}
    id = 0
    for line in labels_detected:
        arr = line.split('\t')
        lines,labels = arr[0].split(constants.value_separator), arr[1].split(constants.label_separator)
        data_dic[id] = {'lines':lines, 'labels':labels}
        id+=1
    return data_dic

def populate_graph():
    data_dic = read_labels_detected()
    gen_graph(data_dic)
