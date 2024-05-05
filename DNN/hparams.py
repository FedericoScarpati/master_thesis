import numpy as np
seed = 42
#split_ratio = [0.01, 0.01, 0.9]
split_ratio = [0.4, 0.4, 0.2]

TARGET_C_LIST = [12.0, 12.25, 12.5, 12.75, 13.0, 13.25, 13.5, 13.75, 14.0, 14.25, 14.5, 14.75, 15.0, 15.25, 15.5, 15.75, 16.0, 16.25, 16.5, 16.75, 17.0, ] 
N_nodes = 50
dataset_name = f'G_list_N{N_nodes}_copy.pt'
#dataset_name = 'N{N_nodes}rand_COL.pt'

TEST_ONLY = False
BEST_N = 5
ITERATE_BEST_N = True
PLANTED_OVERLAP = False

ALPHAVEC = [0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
BETAVEC = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
dim_embedding = 5
MAX_EPOCH = 200
lr = 1e-3
BATCH_SIZE = 4
DROPOUT=0.0
NUM_HEADS = 2
HIDDEN_CHANNELS = 16

#net_name = f'random_2_loss_6x3_32hidd_transfALPHA0.95_BETA0.4_lr0.001'
#net_name = f'SAGE2_loss_6x{HIDDEN_CHANNELS}_D{DROPOUT}'
#net_name = f'planted_2_loss_6x{NUM_HEADS}_{HIDDEN_CHANNELS}hidd_transfALPHA0.95'

net_name = f'{N_nodes}nodes_transformer'
#net_name = f'{N_nodes}nodes_SAGE'
if 'SAGE' in net_name:
    ALPHAVEC.reverse()
    BETAVEC.reverse()
    BATCH_SIZE = 64
    split_ratio = [0.6, 0.2, 0.2]