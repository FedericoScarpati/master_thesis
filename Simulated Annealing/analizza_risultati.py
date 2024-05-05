import pandas as pd
import numpy as np

# Carica i dati dal CSV
df_simann = pd.read_csv('simann_log.txt')
df_continue = pd.read_csv('simann_continue_log.txt')

# Definisce una funzione per calcolare le statistiche richieste per gruppo
def analizza(gruppo):
    N = gruppo['N'].iloc[0]  # Presumendo che N sia lo stesso per tutte le righe con la stessa c
    c = gruppo['c'].iloc[0]
    M = N * c / 2
    
    # Calcola la frazione di numUnsat == 0
    frazione_numUnsat_zero = (gruppo['numUnsat'] == 0).mean()
    
    # Filtra i valori di numUnsat != 0, calcola numUnsat/M e poi media e deviazione standard
    validi = gruppo[gruppo['numUnsat'] != 0]['numUnsat']
    energia_media = (validi / M).mean()
    deviazione_standard_energia = (validi / M).std()
    
    # Calcola media e deviazione standard di nIter
    validi = gruppo[gruppo['numUnsat'] == 0]['nIter']
    nIter_media = validi.mean()
    nIter_deviazione_standard = validi.std()
    
    return pd.Series([N, c, frazione_numUnsat_zero, energia_media, deviazione_standard_energia, nIter_media, nIter_deviazione_standard], 
                     index=['N', 'c', 'frazione_zero', 'energia_media', 'deviazione_standard_energia', 'nIter_media', 'nIter_deviazione_standard'])

# Raggruppa per 'N' e 'c' e applica la funzione di analisi
risultati_simann = df_simann.groupby(['N', 'c']).apply(analizza).reset_index(drop=True)
risultati_continue = df_continue.groupby(['N', 'c']).apply(analizza).reset_index(drop=True)

# Salva i risultati in un nuovo file CSV
risultati_simann.to_csv(f'analisi_simann.csv', mode='a', index=False)
risultati_continue.to_csv(f'analisi_continue.csv', mode='a', index=False)