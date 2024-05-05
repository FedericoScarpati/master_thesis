import subprocess
import os
import itertools
import numpy as np
import shutil
from tqdm import tqdm

NUM_TRIALS = 3
num_nodes_list = [500]
initial_temp = 1.0
initial_temp_continue = round(initial_temp *0.6 , 3)
max_iter = 5000

#c_list = np.arange(10.0, 12, 0.25).tolist()
c_list = [12]
temp_files = ['generated_graphs.txt', 'graph_colors.txt', 'continue_numUnsat_log.txt', 'numUnsat_log.txt']

# Esegui NUM_TRIALS simulazioni per ogni combinazione di num_nodes e c
for num_nodes in num_nodes_list:
    if num_nodes > 1000:
        max_iter = 10000
    for c in c_list:
        print(f"N={num_nodes}, c={c}")

        # Simulated Annealing
        for _ in tqdm(range(NUM_TRIALS)):
            subprocess.run(["./simann", str(num_nodes), str(c), str(initial_temp), str(max_iter)], capture_output=True, text=True)
        
        # Preprocess + Simulated Annealing
        os.system("python3 preprocess.py")
        for i in tqdm(range(NUM_TRIALS)):
            result = subprocess.run(["./simann_continue", str(num_nodes), str(c), str(initial_temp_continue), str(max_iter), str(i)], capture_output=True, text=True)

        quit()
        # Spostare i file temporanei
        base_dir = 'runs'
        target_dir = os.path.join(base_dir, f'N{num_nodes}', f'c{c}')
        os.makedirs(target_dir, exist_ok=True)
        # Sposta ogni file temporaneo nella directory di destinazione
        for temp_file in temp_files:
            source_path = temp_file
            target_path = os.path.join(target_dir, temp_file)
            
            # Controlla se il file esiste prima di tentare di spostarlo
            if os.path.exists(source_path):
                shutil.move(source_path, target_path)
                #print(f"File {source_path} spostato in {target_path}.")
            else:
                print(f"File {source_path} non trovato, non è stato spostato.") 

    # Analizza risultati a N fissato
    os.system("python3 analizza_risultati.py")

    # Spostare i file temporanei
    base_dir = 'runs'
    target_dir = os.path.join(base_dir, f'N{num_nodes}')
    os.makedirs(target_dir, exist_ok=True)
    # Sposta ogni file temporaneo nella directory di destinazione
    temp_files = ['analisi_simann.csv', 'analisi_continue.csv', 'simann_log.txt', 'simann_continue_log.txt']
    for temp_file in temp_files:
        source_path = temp_file
        target_path = os.path.join(target_dir, temp_file)
            
            # Controlla se il file esiste prima di tentare di spostarlo
        if os.path.exists(source_path):
            shutil.move(source_path, target_path)
            #print(f"File {source_path} spostato in {target_path}.")
        else:
            print(f"File {source_path} non trovato, non è stato spostato.") 

print('Finito!')