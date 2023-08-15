import numpy as np
import random
import json
import gc
import sys
import ray
from sklearn.model_selection import train_test_split
from Bio import SeqIO
random.seed(10)
ray.init()
print("Loading TEDB.fasta and filtered_negative.fasta")
sys.stdout.flush()
rec_TE = list(SeqIO.parse("data/TEDB.fasta", "fasta"))
rec_TE = {idx: rec_TE[idx] for idx in list(range(len(rec_TE)))}
gc.collect()
rec_non = list(SeqIO.parse("data/filtered_negative.fasta", "fasta"))
gc.collect()
index_negatives = [x for x in range(0, len(rec_non))]
dicc_size = {0: 2000, 1: 2000, 2: 4000, 3: 1000, 4: 2000, 5: 3000, 6: 15000, 7: 18000}
dicc_dom = {'LTR': 5, 'GAG': 1, 'PROT': 3, 'RT': 0, 'INT': 2, 'RH': 4, 'internal': 6, 'te': 7}
dicc_sf = {'copia': 0, 'gypsy': 1}
dicc = {'A': 0, 'a': 0, 'C': 1, 'c': 1, 'G': 2, 'g': 2, 'T': 3, 't': 3}
print("Fasta files loaded")
sys.stdout.flush()
num_cores = 50

def exploratory_analysis():
    print("Executing exploratory_analysis")
    sys.stdout.flush()
    dicc_analysis = {'TE': [], 'TE2': [], 'PROT': [], 'RT': [], 'INT': [], 'RH': [],
                     'GAG': [], 'LTR': [], 'internal': []}
    for i in rec_TE.values():
        dicc_tmp_dom = json.loads(i.description.split("#")[-1].replace('\'', '\"'))
        dicc_tmp_te = json.loads(i.description.split("#")[-2].replace('\'', '\"'))
        dicc_analysis['TE'].append(dicc_tmp_te['TE'][1]-dicc_tmp_te['TE'][0]+1)
        dicc_analysis['TE2'].append(len(i.seq))
        internal_list = []
        for key, value in dicc_tmp_dom.items():
            if key != 'LTR':
                internal_list += value
            if len(value) == 2:
                dicc_analysis[key].append(value[-1]-value[0]+1)
            else:
                dicc_analysis[key].append(value[1]-value[0]+1)
                dicc_analysis[key].append(value[3]-value[2]+1)
        try:
            dicc_tmp_dom['LTR']
        except Exception:  # pylint: disable=broad-except
            print(f'{i.description}')
            sys.stdout.flush()
        start_internal = min(internal_list)
        end_internal = max(internal_list)
        dicc_analysis['internal'].append(end_internal-start_internal+1)

    print(f"The total amount of RT is {len(dicc_analysis['RT'])}")
    print(f"The total amount of GAG is {len(dicc_analysis['GAG'])}")
    print(f"The total amount of INT is {len(dicc_analysis['INT'])}")
    print(f"The total amount of PROT is {len(dicc_analysis['PROT'])}")
    print(f"The total amount of RH is {len(dicc_analysis['RH'])}")
    print(f"The total amount of LTR is {len(dicc_analysis['LTR'])}")

    print(f"The maximum size of RT is  {max(dicc_analysis['RT'])}")
    print(f"The maximum size of GAG is  {max(dicc_analysis['GAG'])}")
    print(f"The maximum size of INT is {max(dicc_analysis['INT'])}")
    print(f"The maximum size of PROT is {max(dicc_analysis['PROT'])}")
    print(f"The maximum size of RH is {max(dicc_analysis['RH'])}")
    print(f"The maximum size of LTR is {max(dicc_analysis['LTR'])}")

    print(f"The minimum size of RT is {min(dicc_analysis['RT'])}")
    print(f"The minimum size of GAG is {min(dicc_analysis['GAG'])}")
    print(f"The minimum size of INT is {min(dicc_analysis['INT'])}")
    print(f"The minimum size of PROT is {min(dicc_analysis['PROT'])}")
    print(f"The minimum size of RH is {min(dicc_analysis['RH'])}")
    print(f"The minimum size of LTR is {min(dicc_analysis['LTR'])}")

    print(f"The maximum size of internal is {max(dicc_analysis['internal'])}")
    print(f"The minimum size of internal is {min(dicc_analysis['internal'])}")

    print(f"The maximum size of TE is {max(dicc_analysis['TE'])}")
    print(f"The minimum size of TE according to the description is {min(dicc_analysis['TE'])}")
    print(f"The minimum size of TE according to len is {min(dicc_analysis['TE2'])}")
    print(f"The total amount of  TE is {len(rec_TE)}")
    sys.stdout.flush()


def Seq_to_2D(DNA_str):
    longitud_DNA = len(DNA_str)
    Rep_2D = np.zeros((1, 5, longitud_DNA), dtype=np.int8)
    for i in range(longitud_DNA):
        try:
            pos = dicc[DNA_str[i]]
            Rep_2D[0, pos, i] = 1
        except Exception:  # pylint: disable=broad-except
            Rep_2D[0, 4, i] = 1
    return Rep_2D


def sequence_generation(k):
    global index_negatives
    A = ''
    lon = len(A)
    while lon < k:
        azar = random.randint(0, len(index_negatives)-1)
        piece = str(rec_non[index_negatives[azar]].seq)
        if len(A)+len(piece) > k:
            faltante = k-len(A)
            A = A+piece[0:faltante]
        else:
            A = A+str(rec_non[index_negatives[azar]].seq)
        lon = len(A)
        # index_negatives.remove(index_negatives[azar])
    return A


def extend_DNA_with_negatives(DNA_sequence, background, start_bg):
    size_start = random.randint(1500, 2000)
    size_end = random.randint(1500, 2000)
    size_total_partial = size_start+len(DNA_sequence)+size_end
    size_total = (int(size_total_partial/100)+1)*100
    size_end = size_total-size_start-len(DNA_sequence)
    A = background[start_bg:start_bg+size_start]
    B = background[start_bg+len(A)+len(DNA_sequence):start_bg+len(A)+len(DNA_sequence)+size_end]
    DNA_new = A+DNA_sequence+B
    return DNA_new, size_start


def DNA_dataset(item, background, start_bg):
    DNA = str(item.seq)
    DNA, long_start_negative = extend_DNA_with_negatives(DNA, background, start_bg)
    ventana = len(DNA)
    DNA_2D = Seq_to_2D(DNA)
    label = np.zeros((1, 3, int(ventana/100), 11), dtype=np.float32)
    dicc_tmp_dom = json.loads(item.description.split('#')[-1].replace('\'', '\"'))
    dicc_tmp_te = json.loads(item.description.split('#')[-2].replace('\'', '\"'))
    pto_ref = dicc_tmp_te['TE'][0]
    internal_list = []
    for key, value in dicc_tmp_dom.items():
        if key != 'LTR':
            internal_list += value
        lista = dicc_tmp_dom[key]
        if len(lista) == 4:
            indice = [lista[0]-pto_ref+long_start_negative, lista[2]-pto_ref+long_start_negative]
        elif len(lista) == 2:
            indice = [lista[0]-pto_ref+long_start_negative]
        dom = dicc_dom[key]+3
        size_dom = dicc_size[dicc_dom[key]]
        k = 0
        for i in indice:
            label[0, 0, int(i/100), 0] = 1
            label[0, 0, int(i/100), 1] = (i-int(i/100)*100)/100
            label[0, 0, int(i/100), 2] = (lista[k*2+1]-lista[k*2])/size_dom
            label[0, 0, int(i/100), dom] = 1
            try:
                sf = dicc_sf[item.id.split('#')[2]]+9
                label[0, 0, int(i/100), sf] = 1
            except Exception:  # pylint: disable=broad-except
                pass
            k = k+1
    del dicc_tmp_dom['LTR']
    start_internal = min(internal_list)
    end_internal = max(internal_list)
    start_te = dicc_tmp_te['TE'][0]
    end_te = dicc_tmp_te['TE'][1]

    i = start_internal - pto_ref
    lon = end_internal - start_internal + 1
    size_dom = dicc_size[dicc_dom['internal']]
    label[0, 1, int(i/100), 0] = 1
    label[0, 1, int(i/100), 1] = (i-int(i/100)*100)/100
    label[0, 1, int(i/100), 2] = lon/size_dom
    try:
        sf = dicc_sf[item.id.split('#')[2]]+3
        label[0, 1, int(i/100), sf] = 1
    except Exception:  # pylint: disable=broad-except
        pass

    i = start_te - pto_ref
    lon = end_te - start_te + 1
    size_dom = dicc_size[dicc_dom['te']]
    label[0, 2, int(i/100), 0] = 1
    label[0, 2, int(i/100), 1] = (i-int(i/100)*100)/100
    label[0, 2, int(i/100), 2] = lon/size_dom
    try:
        sf = dicc_sf[item.id.split('#')[2]]+3
        label[0, 2, int(i/100), sf] = 1
    except Exception:  # pylint: disable=broad-except
        pass
    return DNA_2D, label

def complete_with_negatives(Rep2D, Label, ventana, num_labels, background):
    size_faltante = ventana-Rep2D.shape[2]
    #A = sequence_generation(size_faltante)
    A = background[-size_faltante:]
    DNA_faltante = Seq_to_2D(A)
    label = np.zeros((1, 3, int(size_faltante/100), num_labels), dtype=np.float32)
    return np.append(Rep2D, DNA_faltante, axis=2), np.append(Label, label, axis=2)

@ray.remote
def row_creation(rec_TE, background, list_indexes, ventana, num_labels):
    flag = 'first_in_seq'
    cont_seq = 0
    Rep2D = np.zeros((1,1,0))
    gc.collect()
    for i, idx in enumerate(list_indexes):
        gc.collect()
        item = rec_TE[idx]
        Rep2D_single, Label_single = DNA_dataset(item, background, Rep2D.shape[2])
        if flag == 'first_in_seq':
            Rep2D = Rep2D_single
            Label = Label_single
            flag = 'other_in_seq'
            Z_vector = np.zeros((1, 10), dtype=np.int64)
            Z_vector[0, cont_seq] = idx
            cont_seq += 1
        elif flag == 'other_in_seq':
            long_base = Rep2D.shape[2]
            long_anexo = Rep2D_single.shape[2]
            if long_base == ventana:
                gc.collect()
                break
            elif (long_base+long_anexo > ventana) or (i == (len(list_indexes)-1)):
                Rep2D, Label = complete_with_negatives(Rep2D, Label, ventana, num_labels, background)
                gc.collect()
                break
            else:
                Rep2D = np.append(Rep2D, Rep2D_single, axis=2)
                Label = np.append(Label, Label_single, axis=2)
                Z_vector[0, cont_seq] = idx
                cont_seq += 1
    gc.collect()
    return [Rep2D, Label, Z_vector, list_indexes[i-1:]]


def optimized_dataset_creation():
    print("Executing optimized_dataset_creation")
    sys.stdout.flush()
    opcion = '/shared/home/sorozcoarias/coffea_genomes/Simon/YORO/data'
    step = 5000
    ventana = 50000
    num_labels = 11
    indexes_list = list(range(len(rec_TE)))
    flag = True
    num_tes_per_row = 10
    X_data = np.zeros((0,5,ventana),dtype=np.int8)
    Y_data = np.zeros((0,3,500,num_labels),dtype=np.float32)
    Z_data = np.zeros((0,10),dtype=np.int64)
    cont = 0
    while flag == True:
        row_list = []
        for i in range(num_cores):
            indexes_for_row = []
            len_ind = len(indexes_for_row)
            while len_ind < num_tes_per_row:
                if len(indexes_list)==0:
                    break
                number = random.choice(list(range(len(indexes_list))))
                index = indexes_list.pop(number)
                indexes_for_row.append(index)
                dicc_tmp_dom = json.loads(rec_TE[index].description.split('#')[-1].replace('\'', '\"'))
                if 'LTR' not in dicc_tmp_dom:
                    indexes_for_row.pop(-1)
                len_ind = len(indexes_for_row)
            rec_TE_subgroup = {key: rec_TE[key] for key in indexes_for_row if key in rec_TE}
            background = sequence_generation(ventana+10000)
            row_list.append(row_creation.remote(rec_TE_subgroup, background, indexes_for_row, ventana, num_labels))
        ray_result = ray.get(row_list)
        remaining = []
        x_list = []
        y_list = []
        z_list = []
        for x in ray_result:
            x_list.append(x[0])
            y_list.append(x[1])
            z_list.append(x[2])
            remaining = remaining + x[3]
        tupla_x = tuple(x_list)
        tupla_y = tuple(y_list)
        tupla_z = tuple(z_list)
        indexes_list = indexes_list + remaining
        del ray_result
        del x_list
        del y_list
        del z_list
        gc.collect()
        empty_x = np.zeros((num_cores,5,ventana), dtype=np.int8)
        empty_y = np.zeros((num_cores,3,int(ventana/100),num_labels), dtype=np.float32)
        empty_z = np.zeros((num_cores,10),dtype=np.int64)
        np.concatenate(tupla_x, out=empty_x, axis=0)
        np.concatenate(tupla_y, out=empty_y, axis=0)
        np.concatenate(tupla_z, out=empty_z, axis=0)
        del tupla_x
        del tupla_y
        del tupla_z
        gc.collect()
        X_data = np.append(X_data, empty_x, axis=0)
        Y_data = np.append(Y_data, empty_y, axis=0)
        Z_data = np.append(Z_data, empty_z, axis=0)
        gc.collect()
        del empty_x
        del empty_y
        del empty_z
        gc.collect()
        print(X_data.shape[0])
        sys.stdout.flush()
        if len(indexes_list) == 0:
            flag = False
        if X_data.shape[0]%step == 0:
            seed = 7
            cont = cont+step
            X_train, X_test_dev, Y_train, Y_test_dev, Z_train, Z_test_dev = train_test_split(X_data, Y_data, Z_data, test_size = 0.2, random_state=seed)
            del X_data
            del Y_data
            del Z_data
            gc.collect()
            X_dev, X_test, Y_dev, Y_test, Z_dev, Z_test = train_test_split(X_test_dev, Y_test_dev, Z_test_dev, test_size=0.5, random_state=seed)
            del X_test_dev
            del Y_test_dev
            del Z_test_dev
            gc.collect()
            print(X_train.shape)
            print(Y_train.shape)
            print(X_dev.shape)
            print(Y_dev.shape)
            print(X_test.shape)
            print(Y_test.shape)
            print('Number of TEs added: '+str(len(rec_TE)-len(indexes_list)))
            print('cont: '+str(cont))
            sys.stdout.flush()
            np.save(f'{opcion}/X_train{cont}',X_train)
            np.save(f'{opcion}/Y_train{cont}',Y_train)
            np.save(f'{opcion}/Z_train{cont}',Z_train)

            np.save(f'{opcion}/X_dev{cont}',X_dev)
            np.save(f'{opcion}/Y_dev{cont}',Y_dev)
            np.save(f'{opcion}/Z_dev{cont}',Z_dev)

            np.save(f'{opcion}/X_test{cont}',X_test)
            np.save(f'{opcion}/Y_test{cont}',Y_test)
            np.save(f'{opcion}/Z_test{cont}',Z_test)
            X_data = np.zeros((0,5,50000),dtype=np.int8)
            Y_data = np.zeros((0,3,500,num_labels),dtype=np.float32)
            Z_data = np.zeros((0,10),dtype=np.int64)
            gc.collect()


def main():
    exploratory_analysis()
    gc.collect()
    optimized_dataset_creation()


if __name__ == "__main__":
    main()
