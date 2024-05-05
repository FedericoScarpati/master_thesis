import os
import torch
import numpy as np
from torch_geometric.utils import to_dense_adj
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt



def train_and_save_model(net_instance, scheduler, optimizer, trainloader, valiloader, max_epochs=20, 
                         embedding_shuffle_epochs = 20, continue_training=False, instance_name = 'prova', 
                         ALPHA = 0.95, BETA = 0.3, G_list = G_list):
    
    net_instance.to(device)
    
    # Load model and continue training
    if continue_training:
        model_filename = f'{instance_name}' + f'.pth'
        model_path = os.path.join(os.getcwd(), 'models', model_filename)
        net_instance.load_state_dict(torch.load(model_path))

    # Initialize history lists
    train_clashes_loss = []
    train_sparse_loss = []
    train_loss = []
    vali_loss = []

    # Best model selection
    best_vali_loss = float('inf')
    best_model = None

    # Training Loop
    for epoch in tqdm(range(max_epochs), desc=f'Training ALPHA={ALPHA} BETA={BETA}', unit='epoch'):
        net_instance.train()

        # Contatori epoch
        epoch_loss = 0
        epoch_clashes_loss = 0
        epoch_sparse_loss = 0
        count = 0

        # Training batch
        for batch in trainloader:
            count += 1
            batch.to(device)
            x = batch.x
            edge_index = batch.edge_index

            adj_mat = to_dense_adj(batch.edge_index)

            optimizer.zero_grad()
            mid, out = net_instance(x, edge_index)

            # Intermediate coloring loss
            mid_clash = torch.mul(adj_mat, (mid @ mid.t())).sum() / (batch.num_edges)
            mid_norm = (1 - (mid ** 2).sum() / mid.shape[0])
            mid_loss = ALPHA * mid_clash + (1-ALPHA)*mid_norm

            # Final loss
            out_clash = torch.mul(adj_mat, (out @ out.t())).sum() / (batch.num_edges)
            out_norm = 1 - (out ** 2).sum() / out.shape[0]
            out_loss = ALPHA * out_clash + (1 - ALPHA) * out_norm

            # Combine intermediate and final
            loss = (1-BETA) * mid_loss + BETA * out_loss

            # Add total loss, clashes and norm of only final output
            epoch_clashes_loss += out_clash
            epoch_sparse_loss += out_norm
            epoch_loss += loss

            loss.backward()
            optimizer.step()

        # Normalize loss on batchsize and save to history
        epoch_loss = epoch_loss / count
        epoch_clashes_loss = epoch_clashes_loss / count
        epoch_sparse_loss = epoch_sparse_loss / count

        train_loss.append(epoch_loss.item())
        train_clashes_loss.append(epoch_clashes_loss.item())
        train_sparse_loss.append(epoch_sparse_loss.item())

        # Print update
        if epoch % 10 == 0:
            print('Epoch:', epoch)
            print('Total Loss:', round(epoch_loss.item(), 4))
            print('Clashes:', round(epoch_clashes_loss.item(), 4))
            print('Sparse Loss', round(epoch_sparse_loss.item(), 4))


        # Validation
        net_instance.eval()
        loss_vali_epoch = 0
        count = 0
        for batch in valiloader:
            batch.to(device)
            x = batch.x
            edge_index = batch.edge_index
            count += 1
            adj_mat = to_dense_adj(batch.edge_index)

            mid, out = net_instance(x, edge_index)

            # Intermediate coloring loss
            mid_clash = torch.mul(adj_mat, (mid @ mid.t())).sum() / (batch.num_edges)
            mid_norm = (1 - (mid ** 2).sum() / mid.shape[0])
            mid_loss = ALPHA * mid_clash+ (1-ALPHA)* mid_norm

            # Final loss
            out_clash = torch.mul(adj_mat, (out @ out.t())).sum() / (batch.num_edges)
            out_norm = 1 - (out ** 2).sum() / out.shape[0]
            out_loss = ALPHA * out_clash + (1 - ALPHA) * out_norm

            # Combine intermediate and final
            loss_vali = (1 - BETA) * mid_loss + BETA * out_loss
            loss_vali_epoch += loss_vali

        loss_vali_epoch = loss_vali_epoch / count

        # Save best_model on validation score
        if loss_vali_epoch < best_vali_loss:
            best_model = net_instance.state_dict()
            best_vali_loss = loss_vali_epoch

        vali_loss.append(loss_vali_epoch.item())
        scheduler.step(loss_vali_epoch)

        # Print update
        if epoch % 10 == 0:
            print('Vali Loss: ', round(loss_vali_epoch.item(), 4))

        # Shuffle embedding
        if epoch % embedding_shuffle_epochs == 0 and epoch > 101:
            G_list = random_embedding(G_list)
            for i in range(len(G_list)):
                G_dataset[i].x = G_list[i].x

        # Salva best_model ogni 20 epoche
        if epoch > 1 and epoch % 20 == 0:
            model_filename = f'{instance_name}' + f'.pth'
            model_path = os.path.join(os.getcwd(), 'models', model_filename)
            if not os.path.exists(os.path.join(os.getcwd(), 'models')):
                os.makedirs(os.path.join(os.getcwd(), 'models'))
            torch.save(best_model, model_path)

            # Plot
            plt.scatter(range(len(train_loss)), train_loss, label=f'Train {round(train_loss[-1],4)}', color='red', s=8)
            plt.scatter(range(len(train_clashes_loss)), train_clashes_loss, label=f'Clashes {round(train_clashes_loss[-1],4)}', color='orange', s=8)
            plt.scatter(range(len(vali_loss)), vali_loss, label=f'Validation {round(vali_loss[-1],4)}', color='blue', s=8)
            plt.scatter(range(len(vali_loss)), train_sparse_loss, label=f'One-Hotness {round(train_sparse_loss[-1],4)}', color='brown', s=8)
            plt.title(f'Loss vs epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plot_filename = f'{instance_name}' + f'.png'
            plot_path = os.path.join(os.getcwd(), 'plots', plot_filename)
            if not os.path.exists(os.path.join(os.getcwd(), 'plots')):
                os.makedirs(os.path.join(os.getcwd(), 'plots'))
            plt.savefig(plot_path)
            plt.close()

    # Plot with zoom on last 200 epochs
    nplt = min(max_epochs,200)
    plt.scatter(range(len(train_loss))[-nplt:], train_loss[-nplt:], label=f'Train {round(train_loss[-1],4)}', color='red', s=5)
    plt.scatter(range(len(train_clashes_loss))[-nplt:], train_clashes_loss[-nplt:], label=f'Clashes {round(train_clashes_loss[-1],4)}', color='orange', s=5)
    plt.scatter(range(len(vali_loss))[-nplt:], vali_loss[-nplt:], label=f'Validation {round(vali_loss[-1],4)}', color='blue', s=5)
    plt.scatter(range(len(vali_loss))[-nplt:], train_sparse_loss[-nplt:], label=f'One-Hotness {round(train_sparse_loss[-1],4)}', color='brown', s=5)
    plt.title(f'Loss vs epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    final_plot_filename = f'final_{instance_name}' + f'.png'
    final_plot_path = os.path.join(os.getcwd(), 'plots', final_plot_filename)
    plt.savefig(final_plot_path)


def test_model(net_instance, device, testloader, instance_name ='prova', BEST_N = 1, ITERATE_BEST_N=False):

    model_path = os.path.join(os.getcwd(), 'models', instance_name + f'.pth')

    if not os.path.exists(model_path):
        print('Model not found')
        quit()
        
    # Initialize results dictionary
    results_by_c = {}
    
    # Energy list of computed graphs
    graphs_energies = []
      
    # Load model
    net = net_instance
    net.load_state_dict(torch.load(model_path))
    net.to(device)
    net.eval()
    
    # Test loop
    with torch.no_grad():
        for batch in tqdm(testloader, desc='Testing', unit='batch'):
            batch.to(device)
            for i in range(len(batch)):
                best_acc = 1
                x = batch[i].x
                edge_index = batch[i].edge_index
                c = float(batch[i].c.item())
                adj_mat = to_dense_adj(batch[i].edge_index)
                num_edges = batch[i].num_edges
                
                for _ in range(BEST_N):
                    
                    _ , out = net_instance(x, edge_index)
                    
                    # Compute graph coloring accuracy and loss                  
                    graph_coloring_accuracy = hard_loss_compute(hard_color_assign(out), batch[i].edge_index) / (num_edges)
                    graph_coloring_loss = torch.mul(adj_mat, (out @ out.t())).sum() / (num_edges)
                                       
                    if graph_coloring_accuracy < best_acc:
                        best_acc = graph_coloring_accuracy
                        best_loss = graph_coloring_loss
                    
                    if best_acc < 1e-5:
                        break  
                    
                    # Iterate over the output
                    if ITERATE_BEST_N and BEST_N > 1:
                        x = out
                    
                    # Generate random one hot encoding for the input
                    elif not ITERATE_BEST_N and BEST_N > 1:
                        n, m = x.shape
                        indices = torch.randint(low=0, high=m, size=(n,))
                        one_hot = torch.zeros(n, m).to(device)
                        one_hot[torch.arange(n), indices] = 1
                        one_hot.to(device)
                        x = one_hot 
                
                graph_coloring_accuracy = best_acc
                graph_coloring_loss = best_loss
                
                # Save c and energy in the list to make statistics on it
                graph_dict = {'c': c, 'energy': best_acc}
                graphs_energies.append(graph_dict)
                
                # Add result in results dictionary
                if c not in results_by_c:
                    results_by_c[c] = {'coloring_accuracy': 0.0, 'coloring_loss': 0.0, 'count': 0}
                results_by_c[c]['count'] += 1
                results_by_c[c]['coloring_accuracy'] += graph_coloring_loss.item()
                results_by_c[c]['coloring_loss'] += graph_coloring_accuracy


    # Graph Coloring Statistics, save in csv file
    # % perfect, average and std dev of graph energies grouped by c

    # Separate energy values by c
    c_energy = {}
    num_nodes = batch[0].x.shape[0]

    for graph in graphs_energies:
        c = graph['c']
        if c not in c_energy:
            c_energy[c] = {'perfect_count': 0, 'total_count': 0, 'energies': []}

        if graph['energy'] <= 1e-5:
            c_energy[c]['perfect_count'] += 1
        else:
            c_energy[c]['energies'].append(graph['energy'])

        c_energy[c]['total_count'] += 1

    # Compute mean and std dev
    sorted_c = sorted(c_energy.keys(), key=lambda x: float(x))
    with open(f'N{num_nodes}_stats.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Connettività', 'Percentuale Perfetti', 'Energia Media', 'Deviazione Standard'])

        for c in sorted_c:
            data = c_energy[c]
            perfect_fraction = (data['perfect_count'] / data['total_count'])
            if data['energies']:
                avg_energy = np.mean(data['energies'])
                std_energy = np.std(data['energies'])
            else:
                avg_energy = 0
                std_energy = 0

            writer.writerow([c, perfect_fraction, avg_energy, std_energy])
    
    # Prepare results dictionary
    for c in results_by_c:
        count = results_by_c[c]['count']
        results_by_c[c]['coloring_accuracy'] /= count
        results_by_c[c]['coloring_accuracy'] = round(results_by_c[c]['coloring_accuracy'] ,4)
        results_by_c[c]['coloring_loss'] /= count
        results_by_c[c]['coloring_loss'] = round(results_by_c[c]['coloring_loss'],4)
    sorted_dict = {k: results_by_c[k] for k in sorted(results_by_c)}

    return sorted_dict

