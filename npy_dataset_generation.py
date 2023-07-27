from Bio import SeqIO
import gc
import numpy as np
import random
from sklearn.model_selection import train_test_split

ventana=50000
opcion='./data/'
sample = 15000
step = 5000

cont = 0        # Empieza en el mismo valor donde quedo la vez pasada
idx_initial = 0    # Empieza en valor impreso y se le suma 1

rec_TE = list(SeqIO.parse("data/TEDB.fasta","fasta"))
rec_non = list(SeqIO.parse("data/negative.fasta","fasta"))

dicc_size={0:2000,1:2000,2:4000,3:1000,4:2000,5:3000,6:15000,7:18000}
dicc_dom={'LTR':5,'GAG':1,'PROT':3,'RT':0,'INT':2,'RH':4,'internal-gypsy':6,'internal-copia':7,'te':7}
dicc_sf={'copia':0, 'gypsy':1}

def Seq_to_2D(DNA_str,tipo,ventana):
  longitud_DNA = len(DNA_str)
  if tipo=='dataset':
    comienzo = int((ventana - longitud_DNA)/2)
    Rep_2D = np.zeros((1,5,ventana),dtype=np.int8)
    for i in range(longitud_DNA):
      if DNA_str[i]=='A' or DNA_str[i]=='a':
        Rep_2D[0,0,comienzo+i]=1
      elif DNA_str[i]=='C' or DNA_str[i]=='c':
        Rep_2D[0,1,comienzo+i]=1
      elif DNA_str[i]=='G' or DNA_str[i]=='g':
        Rep_2D[0,2,comienzo+i]=1
      elif DNA_str[i]=='T' or DNA_str[i]=='t':
        Rep_2D[0,3,comienzo+i]=1
      else:
        Rep_2D[0,4,comienzo+i]=1
  elif tipo=='weights':
    comienzo = 0
    Rep_2D = np.zeros((5,longitud_DNA,1,1),dtype=np.int8)
    for i in range(longitud_DNA):
      if DNA_str[i]=='A' or DNA_str[i]=='a':
        Rep_2D[0,comienzo+i,0,0]=1
      elif DNA_str[i]=='C' or DNA_str[i]=='c':
        Rep_2D[1,comienzo+i,0,0]=1
      elif DNA_str[i]=='G' or DNA_str[i]=='g':
        Rep_2D[2,comienzo+i,0,0]=1
      elif DNA_str[i]=='T' or DNA_str[i]=='t':
        Rep_2D[3,comienzo+i,0,0]=1
      else:
        Rep_2D[4,comienzo+i,0,0]=1
  return Rep_2D

def sequence_generation(k):
  global index_negatives
  A=''
  lon=len(A)
  while lon<k:
    azar = random.randint(0,len(index_negatives)-1)
    piece = str(rec_non[index_negatives[azar]].seq)
    if len(A)+len(piece)>k:
      faltante = k-len(A)
      A=A+piece[0:faltante]
    else:
      A=A+str(rec_non[index_negatives[azar]].seq)
    lon=len(A)
    index_negatives.remove(index_negatives[azar])
  return A

def extend_DNA_with_negatives(DNA_sequence):
  size_start = random.randint(1500,2000)
  size_end = random.randint(1500,2000)
  size_total_partial = size_start+len(DNA_sequence)+size_end
  size_total = (int(size_total_partial/100)+1)*100
  size_end = size_total-size_start-len(DNA_sequence)
  A = sequence_generation(size_start)
  B = sequence_generation(size_end)
  DNA_new=A+DNA_sequence+B
  return DNA_new, size_start

def DNA_dataset(item):
  DNA = str(item.seq)
  DNA,long_start_negative = extend_DNA_with_negatives(DNA)
  ventana = len(DNA)
  DNA_2D = Seq_to_2D(DNA,'dataset',ventana)
  label = np.zeros((1,3,int(ventana/100),11),dtype=np.float32)
  dicc_tmp_dom = json.loads(item.description.split('#')[-1].replace('\'','\"'))
  dicc_tmp_te = json.loads(item.description.split('#')[-2].replace('\'','\"'))
  pto_ref = dicc_tmp_te['TE'][0]
  internal_list=[]
  for key, value in dicc_tmp_dom.items():
    if key != 'LTR':
      internal_list+=value
    lista = dicc_tmp_dom[key] 
    if len(lista)==4:
      indice = [lista[0]-pto_ref+long_start_negative, lista[2]-pto_ref+long_start_negative]
    elif len(lista)==2:
      indice = [lista[0]-pto_ref+long_start_negative]
    dom = dicc_dom[key]+3
    size_dom = dicc_size[dicc_dom[key]]
    k=0
    for i in indice:
      label[0,0,int(i/100),0] = 1
      label[0,0,int(i/100),1] = (i-int(i/100)*100)/100
      label[0,0,int(i/100),2] = (lista[k*2+1]-lista[k*2])/size_dom
      label[0,0,int(i/100),dom] = 1
      try:
        sf = dicc_sf[item.id.split('#')[2]]+9
        label[0,0,int(i/100),sf] = 1
      except:
        pass
      k=k+1
  del dicc_tmp_dom['LTR']
  start_internal = min(internal_list)
  end_internal = max(internal_list)
  start_te = dicc_tmp_te['TE'][0]
  end_te = dicc_tmp_te['TE'][1]

  i = start_internal - pto_ref
  lon = end_internal - start_internal + 1
  size_dom = dicc_size[dicc_dom['internal']]
  label[0,1,int(i/100),0] = 1
  label[0,1,int(i/100),1] = (i-int(i/100)*100)/100
  label[0,1,int(i/100),2] = lon/size_dom
  try:
    sf = dicc_sf[item.id.split('#')[2]]+3
    label[0,1,int(i/100),sf] = 1
  except:
    pass

  i = start_te - pto_ref
  lon = end_te - start_te + 1
  size_dom = dicc_size[dicc_dom['te']]
  label[0,2,int(i/100),0] = 1
  label[0,2,int(i/100),1] = (i-int(i/100)*100)/100
  label[0,2,int(i/100),2] = lon/size_dom
  try:
    sf = dicc_sf[item.id.split('#')[2]]+3
    label[0,2,int(i/100),sf] = 1
  except:
    pass
  return DNA_2D,label

def complete_with_negatives(Rep2D,Label,ventana):
  size_faltante = ventana-Rep2D.shape[2]
  A = sequence_generation(size_faltante)
  DNA_faltante = Seq_to_2D(A,'dataset',len(A))
  label = np.zeros((1,3,int(size_faltante/100),11),dtype=np.float32)
  return np.append(Rep2D,DNA_faltante,axis=2),np.append(Label,label,axis=2)

index_negatives = [x for x in range(0,len(rec_non))]

X_data = np.zeros((1,5,ventana),dtype=np.int8)
Y_data = np.zeros((1,3,int(ventana/100),11),dtype=np.float32)
Z_data = np.zeros((1,10),dtype=np.int64)
flag = 'first_in_seq'
flag_save = 0
cont_seq = 0
gc.collect()
for idx in range(idx_initial,len(rec_TE)):
    gc.collect()
    item=rec_TE[idx]
    Rep2D_single,Label_single= DNA_dataset(item)
    if flag == 'first_in_seq':
      Rep2D = Rep2D_single
      Label = Label_single
      flag = 'other_in_seq'
      Z_vector = np.zeros((1,10),dtype=np.int64)
      Z_vector[0,cont_seq]=idx
      cont_seq+=1
    elif flag =='other_in_seq':
      long_base = Rep2D.shape[2]
      long_anexo = Rep2D_single.shape[2]
      if long_base+long_anexo>ventana:
        Rep2D, Label = complete_with_negatives(Rep2D,Label,ventana)
        Y_data = np.append(Y_data, Label,axis=0)
        X_data = np.append(X_data, Rep2D,axis=0)
        Rep2D = Rep2D_single
        Label = Label_single
        cont+=1
        Z_data = np.append(Z_data,Z_vector,axis=0)
        Z_vector = np.zeros((1,10),dtype=np.int64)
        cont_seq = 0
        Z_vector[0,cont_seq]=idx
        cont_seq+=1
        flag_save = 1
        index_negatives = [x for x in range(0,len(rec_non))]
      else:
        Rep2D = np.append(Rep2D,Rep2D_single,axis=2)
        Label = np.append(Label, Label_single, axis = 2)
        Z_vector[0,cont_seq]=idx
        cont_seq+=1
    if (idx%100)==0:
      print(idx)
    if (cont%step==0) and (flag_save==1):
      seed = 7
      X_train, X_test_dev, Y_train, Y_test_dev, Z_train, Z_test_dev = train_test_split(X_data[1:,:,:], Y_data[1:,:,:,:], Z_data[1:,:], test_size = 0.2, random_state=seed)
      X_dev, X_test, Y_dev, Y_test, Z_dev, Z_test = train_test_split(X_test_dev, Y_test_dev, Z_test_dev, test_size=0.5, random_state=seed)
      print(X_train.shape)
      print(Y_train.shape)
      print(X_dev.shape)
      print(Y_dev.shape)
      print(X_test.shape)
      print(Y_test.shape)
      print('idx: '+str(idx))
      print('cont: '+str(cont))

      np.save(opcion+'/X_train{}.npy'.format(cont),X_train)
      np.save(opcion+'/Y_train{}.npy'.format(cont),Y_train)
      np.save(opcion+'/Z_train{}.npy'.format(cont),Z_train)

      np.save(opcion+'/X_dev{}.npy'.format(cont),X_dev)
      np.save(opcion+'/Y_dev{}.npy'.format(cont),Y_dev)
      np.save(opcion+'/Z_dev{}.npy'.format(cont),Z_dev)

      np.save(opcion+'/X_test{}.npy'.format(cont),X_test)
      np.save(opcion+'/Y_test{}.npy'.format(cont),Y_test)
      np.save(opcion+'/Z_test{}.npy'.format(cont),Z_test)

      X_data = np.zeros((1,5,ventana),dtype=np.int8)
      Y_data = np.zeros((1,3,int(ventana/100),11),dtype=np.float32)
      Z_data = np.zeros((1,10),dtype=np.int64)
      flag_save = 0
      gc.collect()
    if cont==sample:
      break
print(cont)

