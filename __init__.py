import filenames,constants,os
import generateData
import generatePlotReducedDendro
import generate_cluster_labels
import populateGraph
import generateDendrogram

if __name__ == '__main__':
    output_file_prefix = 'c'
    #generateData.generate_data(input_file=filenames.sentences_file,separator=constants.empty_separator, output_file_prefix = output_file_prefix)
    #os.system(constants.hier_code_root+"/hier "+output_file_prefix+'wvec.dat dummy.txt 1 > logs.txt')
    #clusters, rev_label, X = generatePlotReducedDendro.generate(output_file_prefix)
    #generate_cluster_labels.generate_labels()
    #populateGraph.populate_graph()
    generateDendrogram.generate_dendrogram()