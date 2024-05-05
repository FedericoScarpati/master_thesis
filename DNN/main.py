import os
import torch
from torch.utils.data import random_split
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj
from tqdm import tqdm
import numpy as np
from tabulate import tabulate

from hparams import *
from my_functions import *
from GNNs import *
from training_functions import *


# Set seeds for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('CUDA available')
else:
    device = torch.device('cpu')
    print('CPU available')

datadir = os.getcwd()
print('Current directory: ', datadir)
if not os.path.exists('models'):
    os.makedirs('models')


# Load dataset
G_list = torch.load(dataset_name)
G_list = random_embedding(G_list)
G_dataset = my_PyG_dataset(G_list)
G_dataset.print_summary()

# Dataloader
train_size = int(G_dataset.len() * split_ratio[0])
vali_size = int(G_dataset.len() * split_ratio[1])
test_size = G_dataset.len() - (train_size + vali_size)

train_set, vali_set, test_set = random_split(G_dataset, [train_size, vali_size, test_size])

BATCH_SIZE = BATCH_SIZE
trainloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
valiloader = DataLoader(vali_set, batch_size=BATCH_SIZE, shuffle=True)
testloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

# Training loop
for ALPHA, BETA in zip(ALPHAVEC, BETAVEC):
    print(f'ALPHA: {ALPHA}, BETA: {BETA}')

    instance_name = net_name + f'ALPHA{ALPHA}_BETA{BETA}' 
    print(instance_name)

    # Select network instance
    if 'transformer' in net_name:
        net_instance = Coloring_Transformer(5, HIDDEN_CHANNELS, NUM_HEADS)
    else:
        net_instance = Coloring_GraphSAGE(5, HIDDEN_CHANNELS, DROPOUT)
    print(net_instance)
    
    optimizer = torch.optim.Adam(net_instance.parameters(), lr,  betas = (momentum_value, 0.8))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, cooldown=50, patience=600, verbose=True)

    # Training
    if not TEST_ONLY: train_and_save_model(net_instance, scheduler, optimizer, trainloader, valiloader,
                         max_epochs=MAX_EPOCH, instance_name=instance_name, ALPHA = ALPHA, BETA=BETA)

    # Test
    results_dict = test_model(net_instance, device, testloader, instance_name=instance_name,
                              BEST_N=BEST_N, ITERATE_BEST_N=ITERATE_BEST_N)

    # *** Test results processing ***
    # Make a pretty table of results
    tabella = [(chiave, results_dict[chiave]['coloring_loss'], results_dict[chiave]['coloring_accuracy'], results_dict[chiave]['overlap'],
                results_dict[chiave]['count']) for chiave in results_dict]
    intestazioni_colonne = ['C', 'Coloring Loss', 'Coloring Accuracy (err)', 'Overlap','# Graphs']

    # Average accuracy of the whole set
    weighted_accuracy = sum([results_dict[chiave]['coloring_accuracy'] * results_dict[chiave]['count'] for chiave in results_dict]) / sum([results_dict[chiave]['count'] for chiave in results_dict])

    # Print table
    print(tabulate(tabella, headers=intestazioni_colonne, tablefmt='pretty'))
    print(f'Weighted average accuracy: {round(weighted_accuracy, 4)}')
    
    # Log file for the instance
    results_file_path = os.path.join('models', instance_name +f'_log.txt')
    with open(results_file_path, 'a') as results_file:
        results_file.write(f'N{N_nodes} Graph Coloring Results\n')
        results_file.write(tabulate(tabella, headers=intestazioni_colonne, tablefmt='plain'))
        results_file.write(f'\nWeighted average accuracy: {round(weighted_accuracy, 4)}')
        results_file.write(f'\nALPHA:{ALPHA}')
        results_file.write(f'\nBETA:{BETA}')
        results_file.write(f'\nIterate output:{ITERATE_BEST_N}')
        results_file.write(f'\nBEST_N:{BEST_N}\n')
        results_file.write(get_network_structure(net_instance))
        results_file.write(f'\nParameters:{count_parameters(net_instance)}\n-------\n')
        results_file.close()
    
    # Log file for architecture synthesis
    summary_file_path = os.path.join('models', net_name + '_summary.txt')
    with open(summary_file_path, 'a') as summary_file:
        summary_file.write(f'ALPHA: {ALPHA}\n')
        summary_file.write(f'BETA: {BETA}\n')
        summary_file.write(f'Weighted average accuracy: {round(weighted_accuracy, 4)}\n')
        summary_file.write(f'Parameters:{count_parameters(net_instance)}\n-------\n')
        summary_file.close()    