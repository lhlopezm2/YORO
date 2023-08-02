from Bio import SeqIO
import os
import pandas as pd
import numpy as np
import requests
import sys
from auxiliary_functions.Web_genome import download


def find(name, path):
    """Returns the path of the gff file for a given a species
    name and the path where the gffs are located """
    for root, _, files in os.walk(path, topdown=True):
        if name in files:
            return os.path.join(root, name)
    return None


def real_size(row):
    """Returns the size in MB of the genomes to be downloaded"""
    try:
        headers = requests.get(row['Filtered data'], stream=True, timeout=5).headers
        number = headers.get("Content-length")
        if number is None:
            number = 0
    except Exception:  # pylint: disable=broad-except
        number = 0
    return float(number)/(1024**2)


def genoma_dicc(file):
    """Returns a dictionary using the id as the key and the sequence as the value"""
    parser = list(SeqIO.parse(file, 'fasta'))
    if ' ' in parser[0].id:
        genoma = [(x.id.split(' ')[0], x.seq) for x in parser]
    else:
        genoma = [(x.id, x.seq) for x in parser]
    genoma = {x: y for (x, y) in genoma}
    return genoma


def te_extraction(fasta, gff, te_file):
    """Writes a fasta file with only the TE annotated in the gffs"""
    genome = genoma_dicc(fasta)
    df_gff = pd.read_csv(gff, sep='\t', dtype={'Lineages': 'str', 'Divergence': 'str'})
    df_grouped = df_gff.groupby('LTR_ID')
    with open(te_file, 'a', encoding="utf-8") as myfile:
        for _, data in df_grouped:
            data_dicc = extract_data(data)
            try:
                _ = genome[data_dicc["chrom"][0]]
            except Exception:  # pylint: disable=broad-except
                print(f"Genoma {gff} with Chromosome column {data_dicc['chrom'][0]} is not matching chromosome id in fasta")
                sys.stdout.flush()
                break
            domains = {}
            te_dicc = {'TE': [data_dicc["start"][0], data_dicc["end"][-1]]}
            dicc = {'intact_5ltr': 'LTR', 'RH': 'RH', 'RT': 'RT', 'INT': 'INT', 'PROT': 'PROT',
                    'GAG': 'GAG', 'intact_3ltr': 'LTR'}
            for i, start_i in enumerate(data_dicc["start"]):
                try:
                    if dicc[data_dicc["dom"][i]] in domains:
                        domains[dicc[data_dicc["dom"][i]]] += [start_i, data_dicc["end"][i]]
                    else:
                        domains[dicc[data_dicc["dom"][i]]] = [start_i, data_dicc["end"][i]]
                except Exception:  # pylint: disable=broad-except
                    pass
            myfile.write(f">{data_dicc['species'][0]}#{data_dicc['chrom'][0]}#{data_dicc['s_f'][0]}#{str(te_dicc)}#{str(domains)}\n")
            interval = [data_dicc['start'][0], data_dicc['end'][-1]]
            sequence = str(genome[data_dicc['chrom'][0]][interval[0]:interval[1]])
            if len(sequence)==0:
                print("This LTR-RT has zero length, which implies that mapping between gff and fasta is wrong")
                print(f">{data_dicc['species'][0]}#{data_dicc['chrom'][0]}#{data_dicc['s_f'][0]}#{str(te_dicc)}#{str(domains)}\n")
                print(gff)
                print(f"This chromosome has a length of {len(genome[data_dicc['chrom'][0]])}")
                # sys.exit(1)
            myfile.write(sequence+"\n")


def extract_data(data):
    """Extracts start, end, sf, dom, species and chrom from data"""
    data_dicc = {}
    data.sort_values(["Start"], inplace=True)
    data_dicc["start"] = np.array(data["Start"])
    data_dicc["end"] = np.array(data["End"])
    data_dicc["s_f"] = np.array(data["Superfamily"])
    data_dicc["dom"] = np.array(data["Domain"])
    data_dicc["species"] = np.array(data["Species"])
    data_dicc["chrom"] = np.array(data["Chromosome"])
    return data_dicc


def obtain_genome_size():
    """Writes a csv file with the real size of the genomes to be downloaded
    and returns it as a pandas obejct"""
    if os.path.exists('./data/dataset.csv'):
        df_dataset = pd.read_csv('./data/dataset.csv')
        print('dataset.csv ya existe')
        sys.stdout.flush()
    else:
        df_dataset = pd.read_csv('data/genomes_links.csv', sep=";")
        df_dataset = df_dataset[['Order', 'Family', 'Species', 'Filtered data', 'genome_size']]
        df_dataset.sort_values(['genome_size'], inplace=True)
        df_dataset['real_size'] = df_dataset.apply(real_size, axis=1)
        print(f"El numero de links es de {df_dataset['real_size'].count()}")
        sys.stdout.flush()
        df_dataset = df_dataset[df_dataset['real_size'] > 0]
        print(f"El numero de links accesibles es de {df_dataset['real_size'].count()}")
        sys.stdout.flush()
        df_dataset.sort_values(by=['real_size'], ascending=True, inplace=True)
        df_dataset.reset_index(inplace=True, drop=True)
        df_dataset.reset_index(inplace=True)
        df_dataset['fasta_name'] = df_dataset.apply(
            lambda row: 'Fasta'+str(row['index'])+'.fasta',
            axis=1
        )
        df_dataset['path_annot'] = df_dataset.apply(
            lambda row: find(row['Species'].replace(' ', '_')+'.txt', 'data/dataset_intact_LTR-RT'),
            axis=1
        )
        df_dataset.to_csv('./data/dataset.csv', index=False)
    print(df_dataset[['Species', 'path_annot']].head(5))
    sys.stdout.flush()
    return df_dataset


def build_te_fasta(df_with_size):
    """Builds the fasta file iterating over all the genomes"""
    current_index = -1
    if os.path.exists("./data/TEDB.fasta"):
        lista = list(SeqIO.parse("./data/TEDB.fasta", 'fasta'))
        name = str(lista[-1].id).split('#', maxsplit=1)[0].replace('_', ' ')
        current_index = df_with_size[df_with_size['Species'] == name]['index'].iloc[0]
    for index, row in df_with_size.iterrows():
        if index > current_index:
            if not os.path.exists(f"{row['fasta_name']}"):
                download(
                    path_save='.',
                    link=row['Filtered data'],
                    name=row['fasta_name'],
                    timeout=100
                )
            if not os.path.exists(f"{row['fasta_name']}"):
                print(f"Genome {row['fasta_name']} could not be downloaded")
                sys.stdout.flush()
                continue
            te_extraction(
                fasta=row['fasta_name'],
                gff=row['path_annot'],
                te_file='/shared/home/sorozcoarias/coffea_genomes/Simon/YORO/data/TEDB.fasta'
            )
            os.remove(row['fasta_name'])
        # if index == 84:
        #    break


def update_species_in_fasta(species, df_with_size):
    for index, row in df_with_size.iterrows():
        if row['Species'] in species:
            if not os.path.exists(f"{row['fasta_name']}"):
                download(
                    path_save='.',
                    link=row['Filtered data'],
                    name=row['fasta_name'],
                    timeout=100
                )
            if not os.path.exists(f"{row['fasta_name']}"):
                print(f"Genome {row['fasta_name']} could not be downloaded")
                sys.stdout.flush()
                continue
            te_extraction(
                fasta=row['fasta_name'],
                gff=row['path_annot'],
                te_file='/shared/home/sorozcoarias/coffea_genomes/Simon/YORO/data/TEDB_curated.fasta'
            )
            os.remove(row['fasta_name'])


def check_species(species, description):
    for specie in species:
        if specie in description:
            return False
    return True


def remove_species_from_fasta(species):
    filtered_sequences = []
    for record in SeqIO.parse("data/TEDB.fasta", "fasta"):
        if check_species(species,record.description):
            filtered_sequences.append(record)
    SeqIO.write(filtered_sequences, "data/TEDB_curated.fasta", "fasta")


def main():
    """Creates the dataset.csv and genomes_link.csv"""
    df_with_size = obtain_genome_size()
    action = "curate"
    if action == "build":
        build_te_fasta(df_with_size)
    elif action == "curate":
        species=["Ipomoea_triloba"]
        remove_species_from_fasta(species)
        species = [x.replace('_',' ') for x in species]
        update_species_in_fasta(species, df_with_size)


if __name__ == "__main__":
    main()
