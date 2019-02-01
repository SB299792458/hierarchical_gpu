<h1>Perform hierarchical clustering on twitter data using GPU :</h1>

This project aims at clustering tweets based on their word2vec embeddings using CUDA framework.


<h3>Steps:</h3> 

generateData - generate word2vec data for cuda hierarchical clustering code

hierarchicalGPU.cu - CUDA code, outputs hierarchical sequences

generatePlotReduceDendro - generates clusters from allotted hierarchical sequences based on threshold

generate_cluster_labels - generate cluster labels based on percentile and frequency

populateGraph - generate graph visualization for basic clusters

generateDendrogram - generate hierarchy for the clusters in graph




<h3>Results:</h3>

![alt text](https://github.com/SB299792458/hierarchical_gpu/blob/master/newplot.png?raw=true)

For full result, download : https://raw.githubusercontent.com/shikhar-b/hierarchical_gpu/master/dendrogram_with_labels.html

Clusters with labels : https://github.com/shikhar-b/hierarchical_gpu/blob/master/levelsclusters.txt

Logs (with performance and clustering results) : https://github.com/shikhar-b/hierarchical_gpu/blob/master/logs.txt
