# master_thesis
Available code from the master thesis work "Tackling the Graph Coloring Problem with Physics-Informed Geometrical Deep Learning".

In this repository I've added a selection of the code produced during the work of my master thesis, which aimed to delevop "physics informed" deep learning methods for the graph coloring problem.

There are 2 folders in this repository:

-DNN: this folder contains the code for training a DNN able to color graphs with fixed chromatic number. This includes the code for generating a dataset of random and planted graphs, as well as a list of DNNs developed and the code able to train and tes the model

-Simulated Annealing: this folder contains the mixed "DNN + Simulated Annealing" algorithm for graph coloring, which colors graphs using a trained DNN and removes the remaining conflicts with a Simulated Annealing postprocessing algorithm. The program developed compares the performances of the Simulated Annealing "working alone" vs. the mixed algorithm.

############

Author email:
scarpati.1645252@studenti.uniroma1.it
