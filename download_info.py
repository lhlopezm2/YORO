from Bio import SeqIO
import os
import pandas as pd
import numpy as np
import requests
from auxiliary_functions.Web_genome import download

# Returns the path of the gff file for a given a species name and the path where the gffs are located
def find(name, path):
  for root, dirs, files in os.walk(path, topdown=True):
    if name in files:
      return os.path.join(root, name)
  return None

# Returns the size in MB of the genomes to be downloaded
def real_size(row):
  try:
    number = requests.get(row['Filtered data'], stream=True, timeout = 5).headers.get("Content-length")
    if number == None:
      number = 0
  except:
    number = 0
  return float(number)/(1024**2)

# Returns a dictionary using the id as the key and the sequence as the value
def genoma_dicc(file):
  parser = list(SeqIO.parse(file, 'fasta'))
  if ' ' in parser[0].id:
    genoma = [(x.id.split(' ')[0], x.seq) for x in parser]
  else:
    genoma = [(x.id, x.seq) for x in parser]
  genoma = {x:y for (x,y) in genoma}
  return genoma

# Writes a fasta file with only the TE annotated in the gffs.
def TE_extraction(fasta, gff, TE_file):
  genome = genoma_dicc(fasta)
  df = pd.read_csv(gff,sep='\t', dtype={'Lineages': 'str', 'Divergence': 'str'})
  df_grouped = df.groupby('LTR_ID')
  with open (TE_file, 'a') as myfile:
    for id, data in df_grouped:
      data.sort_values(['Start'], inplace=True)
      start = np.array(data["Start"])
      end = np.array(data["End"])
      sf = np.array(data["Superfamily"])
      dom = np.array(data["Domain"])
      species = np.array(data["Species"])
      chrom = np.array(data["Chromosome"])
      try:
        a = genome[chrom[0]]
      except:
        print(f"Genoma {gff} with Chromosome column {chrom[0]} is not matching chromosome id in fasta")
        continue
      domains={}
      TE_dicc = {'TE': [start[0],end[-1]]}
      dicc = {'intact_5ltr':'LTR','RH':'RH','RT':'RT','INT':'INT','PROT':'PROT','GAG':'GAG','intact_3ltr':'LTR'}
      for i in range(len(start)):
        try:
          if dicc[dom[i]] in domains:
            domains[dicc[dom[i]]]+=[start[i],end[i]]
          else:
            domains[dicc[dom[i]]]=[start[i],end[i]]
        except:
          pass
      myfile.write(f">{species[0]}#{chrom[0]}#{sf[0]}#{str(TE_dicc)}#{str(domains)}\n")
      myfile.write(str(genome[chrom[0]][start[0]:end[-1]])+"\n")
  return None

# Writes a csv file with the real size of the genomes to be downloaded and returns it as a pandas obejct
def obtain_genome_size():
  if os.path.exists('./data/dataset.csv'):
    df = pd.read_csv('./data/dataset.csv')
    print('dataset.csv ya existe')
  else:
    df = pd.read_csv('data/genomes_links.csv', sep = ";")
    df = df[['Order','Family','Species','Filtered data','genome_size']]
    df.sort_values(['genome_size'] , inplace = True)
    df['real_size']= df.apply(lambda row: real_size(row), axis=1)
    print(f"El numero de links es de {df['real_size'].count()}")
    df = df[df['real_size'] > 0]
    print(f"El numero de links accesibles es de {df['real_size'].count()}")
    df.sort_values(by=['real_size'], ascending=True, inplace = True)
    df.reset_index(inplace = True, drop = True)
    df.reset_index(inplace=True)
    df['fasta_name'] = df.apply(lambda row: 'Fasta'+str(row['index'])+'.fasta', axis=1)
    df['path_annot'] = df.apply(lambda row: find(row['Species'].replace(' ','_')+'.txt', 'data/dataset_intact_LTR-RT'), axis=1)
    df.to_csv('./data/dataset.csv', index = False)
  print(df[['Species','path_annot']].head(5))
  return df

# Builds the fasta file iterating over all the genomes
def build_te_fasta(df):
  current_index = -1
  if os.path.exists(f"./data/TEDB.fasta"):
    lista = list(SeqIO.parse(f"./data/TEDB.fasta", 'fasta'))
    name = str(lista[-1].id).split('#')[0].replace('_',' ')
    current_index = df[df['Species']==name]['index'].iloc[0]
  for index, row in df.iterrows():
    if index > current_index:
      if not os.path.exists(f"{row['fasta_name']}"):
        download(path_save = '.', link = row['Filtered data'],name = row['fasta_name'],timeout = 100)
      TE_extraction(fasta = row['fasta_name'],gff = row['path_annot'], TE_file = 'data/TEDB.fasta')
      os.remove(row['fasta_name'])
    #if index == 1:
    #  break
  return None

df = obtain_genome_size()
build_te_fasta(df)
