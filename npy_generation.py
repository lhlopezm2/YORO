# -*- coding: utf-8 -*-
import numpy as np
import random
import json
import gc
from sklearn.model_selection import train_test_split
from Bio import SeqIO
random.seed(10)

rec_TE = list(SeqIO.parse("data/TEDB.fasta", "fasta"))
rec_non = list(SeqIO.parse("data/negative.fasta", "fasta"))
index_negatives = [x for x in range(0, len(rec_non))]
dicc_size = {0: 2000, 1: 2000, 2: 4000, 3: 1000, 4: 2000, 5: 3000, 6: 15000, 7: 18000}
dicc_dom = {'LTR': 5, 'GAG': 1, 'PROT': 3, 'RT': 0, 'INT': 2, 'RH': 4, 'internal': 6, 'te': 7}
dicc_sf = {'copia': 0, 'gypsy': 1}
dicc = {'A': 0, 'a': 0, 'C': 1, 'c': 1, 'G': 2, 'g': 2, 'T': 3, 't': 3}


def exploratory_analysis():
    dicc_analysis = {'TE': [], 'TE2': [], 'PROT': [], 'RT': [], 'INT': [], 'RH': [],
                     'GAG': [], 'LTR': [], 'internal': []}
    for i in rec_TE:
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
        start_internal = min(internal_list)
        end_internal = max(internal_list)
        dicc_analysis['internal'].append(end_internal-start_internal+1)

    print(f"La cantidad total de RT es de {len(dicc_analysis['RT'])}")
    print(f"La cantidad total de GAG es de {len(dicc_analysis['GAG'])}")
    print(f"La cantidad total de INT es de {len(dicc_analysis['INT'])}")
    print(f"La cantidad total de PROT es de {len(dicc_analysis['PROT'])}")
    print(f"La cantidad total de RH es de {len(dicc_analysis['RH'])}")
    print(f"La cantidad total de LTR es de {len(dicc_analysis['LTR'])}")

    print(f"El tamaño máximo del RT es {max(dicc_analysis['RT'])}")
    print(f"El tamaño máximo del GAG es {max(dicc_analysis['GAG'])}")
    print(f"El tamaño máximo del INT es {max(dicc_analysis['INT'])}")
    print(f"El tamaño máximo del PROT es {max(dicc_analysis['PROT'])}")
    print(f"El tamaño máximo del RH es {max(dicc_analysis['RH'])}")
    print(f"El tamaño máximo del LTR es {max(dicc_analysis['LTR'])}")

    print(f"El tamaño mínimo del RT es {min(dicc_analysis['RT'])}")
    print(f"El tamaño mínimo del GAG es {min(dicc_analysis['GAG'])}")
    print(f"El tamaño mínimo del INT es {min(dicc_analysis['INT'])}")
    print(f"El tamaño mínimo del PROT es {min(dicc_analysis['PROT'])}")
    print(f"El tamaño mínimo del RH es {min(dicc_analysis['RH'])}")
    print(f"El tamaño mínimo del LTR es {min(dicc_analysis['LTR'])}")

    print(f"El tamaño máximo del internal es {max(dicc_analysis['internal'])}")
    print(f"El tamaño mínimo del internal es {min(dicc_analysis['internal'])}")

    print(f"El tamaño máximo del TE es {max(dicc_analysis['TE'])}")
    print(f"El tamaño mínimo del TE es {min(dicc_analysis['TE'])}")
    print(f"El tamaño mínimo del TE es {min(dicc_analysis['TE2'])}")
    print(f"La cantidad total de TE es de {len(rec_TE)}")


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


def extend_DNA_with_negatives(DNA_sequence):
    size_start = random.randint(1500, 2000)
    size_end = random.randint(1500, 2000)
    size_total_partial = size_start+len(DNA_sequence)+size_end
    size_total = (int(size_total_partial/100)+1)*100
    size_end = size_total-size_start-len(DNA_sequence)
    A = sequence_generation(size_start)
    B = sequence_generation(size_end)
    DNA_new = A+DNA_sequence+B
    return DNA_new, size_start


def DNA_dataset(item):
    DNA = str(item.seq)
    DNA, long_start_negative = extend_DNA_with_negatives(DNA)
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


def complete_with_negatives(Rep2D, Label, ventana):
    size_faltante = ventana-Rep2D.shape[2]
    A = sequence_generation(size_faltante)
    DNA_faltante = Seq_to_2D(A)
    label = np.zeros((1, 3, int(size_faltante/100), 11), dtype=np.float32)
    return np.append(Rep2D, DNA_faltante, axis=2), np.append(Label, label, axis=2)


def dataset_creation():
    global index_negatives
    ventana = 50000
    opcion = '.'
    sample = 722084
    step = 5000
    cont = 0        # Empieza en el mismo valor donde quedo la vez pasada
    idx_initial = 0    # Empieza en valor impreso y se le suma 1
    X_data = np.zeros((1, 5, ventana), dtype=np.int8)
    Y_data = np.zeros((1, 3, int(ventana/100), 11), dtype=np.float32)
    Z_data = np.zeros((1, 10), dtype=np.int64)
    flag = 'first_in_seq'
    flag_save = 0
    cont_seq = 0
    gc.collect()
    for idx in range(idx_initial, len(rec_TE)):
        gc.collect()
        item = rec_TE[idx]
        try:
            Rep2D_single, Label_single = DNA_dataset(item)
        except Exception:  # pylint: disable=broad-except
            print(f'La longitud de {item.description} es {len(item.seq)}')
            continue
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
            if long_base+long_anexo > ventana:
                Rep2D, Label = complete_with_negatives(Rep2D, Label, ventana)
                Y_data = np.append(Y_data, Label, axis=0)
                X_data = np.append(X_data, Rep2D, axis=0)
                Rep2D = Rep2D_single
                Label = Label_single
                cont += 1
                Z_data = np.append(Z_data, Z_vector, axis=0)
                Z_vector = np.zeros((1, 10), dtype=np.int64)
                cont_seq = 0
                Z_vector[0, cont_seq] = idx
                cont_seq += 1
                flag_save = 1
                index_negatives = [x for x in range(0, len(rec_non))]
            else:
                Rep2D = np.append(Rep2D, Rep2D_single, axis=2)
                Label = np.append(Label, Label_single, axis=2)
                Z_vector[0, cont_seq] = idx
                cont_seq += 1
        if (idx % 100) == 0:
            print(idx)
        if (cont % step == 0) and (flag_save == 1):
            seed = 7
            X_train, X_test_dev, Y_train, Y_test_dev, Z_train, Z_test_dev = train_test_split(
              X_data[1:, :, :], Y_data[1:, :, :, :], Z_data[1:, :], test_size=0.2, random_state=seed
            )
            X_dev, X_test, Y_dev, Y_test, Z_dev, Z_test = train_test_split(
              X_test_dev, Y_test_dev, Z_test_dev, test_size=0.5, random_state=seed
            )
            print(X_train.shape)
            print(Y_train.shape)
            print(X_dev.shape)
            print(Y_dev.shape)
            print(X_test.shape)
            print(Y_test.shape)
            print('idx: '+str(idx))
            print('cont: '+str(cont))
            np.save(opcion+'/X_train{}.npy'.format(cont), X_train)
            np.save(opcion+'/Y_train{}.npy'.format(cont), Y_train)
            np.save(opcion+'/Z_train{}.npy'.format(cont), Z_train)

            np.save(opcion+'/X_dev{}.npy'.format(cont), X_dev)
            np.save(opcion+'/Y_dev{}.npy'.format(cont), Y_dev)
            np.save(opcion+'/Z_dev{}.npy'.format(cont), Z_dev)

            np.save(opcion+'/X_test{}.npy'.format(cont), X_test)
            np.save(opcion+'/Y_test{}.npy'.format(cont), Y_test)
            np.save(opcion+'/Z_test{}.npy'.format(cont), Z_test)

            X_data = np.zeros((1, 5, ventana), dtype=np.int8)
            Y_data = np.zeros((1, 3, int(ventana/100), 11), dtype=np.float32)
            Z_data = np.zeros((1, 10), dtype=np.int64)
            flag_save = 0
            gc.collect()
        if cont == sample:
            break
    print(cont)
    print(item.description)


def main():
    exploratory_analysis()
    dataset_creation()


if __name__ == "__main__":
    main()
