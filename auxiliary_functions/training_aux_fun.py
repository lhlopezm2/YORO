import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import RMSprop, Adam, Adagrad, SGD, Adadelta, Adamax, Nadam
from tensorflow.keras.layers import Dropout, Activation, Flatten, Concatenate, Dense, Reshape, Add, PReLU, LeakyReLU, BatchNormalization
from tensorflow.keras.regularizers import l2

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.InteractiveSession(config=config)

dicc_size = {0: 2000, 1: 2000, 2: 4000, 3: 1000, 4: 2000, 5: 3000, 6: 15000, 7: 18000}
dicc_sf = {'copia': 0, 'gypsy': 1}
dicc_dom = {'LTR': 5, 'GAG': 1, 'PROT': 3, 'RT': 0, 'INT': 2, 'RH': 4, 'internal': 6, 'te': 7}
ventana = 50000

def loss_domains(y_true, y_pred):
    focus = tf.gather(y_true,tf.constant([0]),axis=-1)
    w1=focus
    w2=(focus-1)*(-1)*(5*5/(500-5*5))
    #w2=(focus-1)*(-1)*(4/(500-4))
    a=1
    b=1
    c=1
    d=1
    weights=tf.concat([(w1+w2)*a,focus*b,focus*c,focus*d, focus*d, focus*d, focus*d, focus*d],axis=-1)
    #weights=tf.concat([(w1+w2)*a,focus*b,focus*c,focus*d, focus*d],axis=-1)
    salida = K.sum(K.pow((y_true-y_pred),2)*weights)
    return salida

def loss_precision_training(y_true, y_pred):
    presence_true = tf.gather(y_true,tf.constant([0]),axis=-1)
    presence_pred = tf.gather(y_pred,tf.constant([0]),axis=-1)
    salida = K.sum(presence_true*presence_pred)/K.sum(presence_pred)
    return salida

def size_log(norm_value):
    return (10**(norm_value/2.405)-1.202)*4000


def nt_region(y,indexes,region):
  sample = y.shape[0]
  ventana = int(y.shape[2]*100)
  nucleotidos = np.zeros((sample,1,ventana))
  valores =[]
  for i in range(sample):
    indices = np.nonzero(y[i,0,:,0])[0]
    for h in range(len(indices)):
        j=indices[h]
        if region == 'dom':
          size = size_log(y[i,0,j,2])
        elif region =='internal':
          size = dicc_size[6]*y[i,0,j,2]
        elif region =='full':
          size = dicc_size[7]*y[i,0,j,2]
        inicio = int(j*100+y[i,0,j,1]*100)
        fin = int(inicio+size)
        nucleotidos[i,0,inicio:fin]=1
  return nucleotidos

def Visualization_LTR(Yhat,Y,indexes,region,opcion,name):
    color={0:'b',1:'r',2:'g',3:'k',4:'y',5:'m',6:'b',7:'r'}

    Yhat_nt = nt_region(Yhat.reshape((1,Yhat.shape[0],Yhat.shape[1],Yhat.shape[2])),indexes,region)
    Y_nt = nt_region(Y.reshape((1,Y.shape[0],Y.shape[1],Y.shape[2])),indexes,region)

    indices = np.nonzero(Y[0,:,0])[0]
    x_dom_true=[]
    colour_dom_true =[]
    for i in indices:
        start_true = int(i*100+Y[0,i,1]*100)
        if region == 'dom':
          key = np.argmax(Y[0,i,indexes[0]:indexes[1]])
          size = size_log(Y[0,i,2])
        elif region=='internal':
          size = dicc_size[6]*Y[0,i,2]
          key = np.argmax(Y[0,i,indexes[0]:indexes[1]])+6
        elif region=='full':
          size = dicc_size[7]*Y[0,i,2]
          key = np.argmax(Y[0,i,indexes[0]:indexes[1]])+6
        x_dom_true.append((start_true, int(size)))
        colour_dom_true.append(color[key])
    y_dom_true=(0.55,0.45)
    colour_dom_true=tuple(colour_dom_true)
        
    indices = np.nonzero(Yhat[0,:,0])[0]
    x_dom_pred=[]
    colour_dom_pred =[]
    for i in indices:
        start_pred = int(i*100+Yhat[0,i,1]*100)
        if region == 'dom':
          key = np.argmax(Y[0,i,indexes[0]:indexes[1]])
          size = size_log(Yhat[0,i,2])
        elif region=='internal':
          size = dicc_size[6]*Yhat[0,i,2]
          key = np.argmax(Y[0,i,indexes[0]:indexes[1]])+6
        elif region=='full':
          size = dicc_size[7]*Yhat[0,i,2]
          key = np.argmax(Y[0,i,indexes[0]:indexes[1]])+6
        x_dom_pred.append((start_pred, int(size)))
        colour_dom_pred.append(color[key])   
    y_dom_pred = (1.5,0.45)
    colour_dom_pred=tuple(colour_dom_pred)

    fig, ax = plt.subplots()
    fig.set_figheight(5)
    fig.set_figwidth(15)
    fig.set_dpi(200)
    ax.broken_barh(x_dom_pred, y_dom_pred, facecolors=colour_dom_pred,alpha=0.7)
    ax.broken_barh(x_dom_true, y_dom_true, facecolors=colour_dom_true,alpha=0.7)
    ax.set_xlim([0, ventana])
    
    if indexes[1]==9 or indexes[1]==8:
      custom_lines = [Line2D([0], [0], color=color[value], lw=4) for key,value in dicc_dom.items() if (value!=6 or value!=7)]
      etiquetas_dom = [key for key in dicc_dom.keys()]
      ax.legend(custom_lines, etiquetas_dom)
    elif indexes[1]==5:
      custom_lines = [Line2D([0], [0], color=color[value+6], lw=4) for key,value in dicc_sf.items()]
      etiquetas_sf = [i for i in dicc_sf.keys()]
      ax.legend(custom_lines, etiquetas_sf)
    plt.savefig(f"{opcion}/visualization{name}.png", format="png")
    return None

def IOU(box1,box2,size1,size2):
    pi1,len1,n1 = box1
    pi2,len2,n2 = box2

    pi1=(pi1+n1)*100
    pf1=pi1+len1*size1
    pi2=(pi2+n2)*100
    pf2=pi2+len2*size2
    xi1 = max([pi1,pi2])
    xi2 = min([pf1,pf2])
    inter_width = xi2-xi1
    inter_area = max([inter_width,0])
    box1_area = len1*size1
    box2_area = len2*size2
    union_area = box1_area+box2_area-inter_area
    iou = inter_area/union_area
    return iou

def NMS(Yhat, threshold_presence, threshold_NMS, indexes, region):
  Yhat_new = np.copy(Yhat)
  for index in range(Yhat.shape[0]):
    mascara = (Yhat[index,:,:,0:1]>=threshold_presence)*1
    data_pred = mascara*Yhat[index,:,:,:]
    data_mod = np.copy(data_pred[0,:,0])
    cont=1
    while cont>0:
      try:
        ind_first = np.nonzero(data_mod)[0][0]
      except:
        break
      ind_nonzero = np.nonzero(data_mod)[0][1:]
      for i in ind_nonzero:
          if region == 'dom':
              box1=[data_pred[0,ind_first,1],1,ind_first]
              box2=[data_pred[0,i,1],1,i]
              size1 = size_log(data_pred[0,ind_first,2])
              size2 = size_log(data_pred[0,i,2])
          else:
              box1=[data_pred[0,ind_first,1],data_pred[0,ind_first,2],ind_first]
              box2=[data_pred[0,i,1],data_pred[0,i,2],i]
              size1=dicc_size[np.argmax(data_pred[0,ind_first,indexes[0]:indexes[1]])]
              size2=dicc_size[np.argmax(data_pred[0,i,indexes[0]:indexes[1]])]
          iou = IOU(box1,box2,size1,size2)
          if iou>=threshold_NMS:
            if data_mod[i]>data_mod[ind_first]:
              data_pred[0,ind_first,:]=0
              data_mod[ind_first]=0
              break
            else:
              data_pred[0,i,:]=0
              data_mod[i]=0
          else:
            data_mod[ind_first]=0
            break
      cont=np.sum(ind_nonzero)
    Yhat_new[index,:,:,:]=data_pred
  return Yhat_new


def metricas_TE(Y_true, Y_pred,indexes,region):
  Y_true_nt = nt_region(Y_true,indexes,region)
  Y_pred_nt = nt_region(Y_pred,indexes,region)
  TP = np.sum(Y_true_nt*Y_pred_nt)
  FP = np.sum(Y_pred_nt)-TP
  FN = np.sum(Y_true_nt)-TP
  TN = Y_true_nt.shape[0]*Y_true_nt.shape[2]-TP-FP-FN
  Precision = TP/(TP+FP)
  Recall = TP/(TP+FN)
  Accuracy = (TP + TN)/ (TP + FN + TN + FP)
  F1 = 2* Precision*Recall/(Precision + Recall)
  A_PR = 0
  A_ROC = 0
  print(f'TP: {TP}')
  print(f'TN: {TN}')
  print(f'FP: {FP}')
  print(f'FN: {FN}')
  print('Precision = {} \n Recall = {} \n Accuracy = {}  \n F1 = {} \n A_PR = {} \n A_ROC = {}'.format(Precision,Recall,Accuracy,F1,A_PR,A_ROC))
  return None

def index_pos(y):
  indices_start=[]
  indices_end=[]
  longitudes=[]
  posiciones = np.absolute(y[1:]-y[0:-1])
  vector = np.nonzero(posiciones)[0]
  
  if len(vector)%2!=0:
    vector=np.append(vector,np.array([49999]))
  for i in range(len(vector)):
    if i%2==0:
      indices_start.append(vector[i])
    else:
      indices_end.append(vector[i])
      longitudes.append(vector[i]-vector[i-1])
  return indices_start,indices_end,longitudes

def iou_parity(box1,box2):
  i1,f1,_=box1
  i2,f2=box2
  xi=max([i1,i2])
  xf = min([f1,f2])
  inter=max([xf-xi,0])
  return inter

def acoplamiento(indices_hat_start,indices_hat_end,indices_true_start,indices_true_end,long_true,include_FN=False):
  dicc={}
  lista=[]
  hat_start,hat_end,true_start,true_end,true_long=([],[],[],[],[])
  for i in range(len(indices_true_start)):
    box1=(indices_true_start[i],indices_true_end[i],long_true[i])
    for j in range(len(indices_hat_start)):
      box2=(indices_hat_start[j],indices_hat_end[j])
      if iou_parity(box1,box2)>0:
        if box1 in dicc:
          dicc[box1].append(box2)
        else:
          dicc[box1]=[box2]
      if j==len(indices_hat_start)-1 and box1 not in dicc:
        if include_FN:
          dicc[box1]=[(0,0)]
        else:
          pass
  for key in dicc.keys():
    for item in dicc[key]:
      lista.append(item)
      hat_start.append(item[0])
      hat_end.append(item[1])
      true_start.append(key[0])
      true_end.append(key[1])
      true_long.append(key[2])
  for j in range(len(indices_hat_start)):
    item=(indices_hat_start[j],indices_hat_end[j])
    if item not in lista:
      hat_start.append(item[0])
      hat_end.append(item[1])
      true_start.append(0)
      true_end.append(0)
      true_long.append(0)
  return hat_start,hat_end,true_start,true_end,true_long

def Plot_parity_TE(Y_true,Yhat_pred,indexes,region,opcion):
  inicio_hat_pred=[]
  inicio_true=[]
  fin_hat_pred=[]
  fin_true=[]
  indices=[]
  longitud=[]
  malos=''
  Y_hat_nt = nt_region(Yhat_pred,indexes,region)
  Y_true_nt = nt_region(Y_true,indexes,region)
  for i in range(Yhat_pred.shape[0]):
    indices_hat_start,indices_hat_end,_ = index_pos(Y_hat_nt[i,0,:])
    indices_true_start,indices_true_end,long_true = index_pos(Y_true_nt[i,0,:])
    indices_hat_start,indices_hat_end,indices_true_start,indices_true_end,long_true=acoplamiento(indices_hat_start,indices_hat_end,indices_true_start,indices_true_end,long_true)
    array_inicio_hat = np.array(indices_hat_start)
    array_inicio_true = np.array(indices_true_start)
    if np.sum((np.absolute(array_inicio_hat-array_inicio_true)>10000)*1)>0:
      malos=malos+str(i)+'-'
    try:
      inicio_hat_pred = inicio_hat_pred + indices_hat_start
      fin_hat_pred = fin_hat_pred + indices_hat_end
    except:
      print('algo salio mal')
      pass
    
    inicio_true = inicio_true + indices_true_start
    fin_true = fin_true + indices_true_end
    longitud = longitud + long_true
    if len(inicio_hat_pred)<len(inicio_true):
      while len(inicio_hat_pred)<len(inicio_true):
        inicio_hat_pred.append(0)
        fin_hat_pred.append(0)
    if len(inicio_hat_pred)>len(inicio_true):
      while len(inicio_hat_pred)>len(inicio_true):
        inicio_true.append(inicio_true[-1])
        fin_true.append(fin_true[-1])
        indices.append(int(len(inicio_true)-1))
        longitud.append(longitud[-1])
  plt.figure()
  plt.scatter(inicio_true, inicio_hat_pred, marker='.', s=10, color='k', linewidths=1)
  plt.plot([0,ventana],[0,ventana],'k',linewidth=0.5)
  plt.plot([400,ventana],[0,ventana-400],'k--',linewidth=0.5)
  plt.plot([0,ventana-400],[400,ventana],'k--',linewidth=0.5)
  plt.xlim([0, ventana])
  plt.ylim([0, ventana])
  plt.xlabel('real')
  plt.ylabel('predicted')
  plt.savefig(f"{opcion}/parity-start.png", format="png")

  start=np.array(inicio_true)
  start_hat=np.array(inicio_hat_pred)
  R2=1-np.sum((start-start_hat)**2)/np.sum((start-np.mean(start))**2)
  print('R^2 = '+str(R2))
  
  plt.figure()
  plt.scatter(fin_true, fin_hat_pred, marker='.', s=10, color='k', linewidths=1)
  plt.plot([0,ventana],[0,ventana],'k',linewidth=0.5)
  plt.plot([400,ventana],[0,ventana-400],'k--',linewidth=0.5)
  plt.plot([0,ventana-400],[400,ventana],'k--',linewidth=0.5)
  plt.xlim([0, ventana])
  plt.ylim([0, ventana])
  plt.xlabel('real')
  plt.ylabel('predicted')
  plt.savefig(f"{opcion}/parity-end.png", format="png")

  A=np.absolute(np.array(inicio_true)-np.array(inicio_hat_pred))
  B=np.array(longitud)
  plt.figure()
  plt.semilogy(B,A,'bo',linewidth=0.5)
  plt.plot([0,25000],[100,100],'--')
  plt.xlim([0, 25000])
  plt.ylim([0,ventana])
  plt.show()
  plt.savefig(f"{opcion}/diff.png", format="png")
  return malos

def Precision_Recall_ROC_TE(y_true,y_hat,indexes,region,opcion):
  Y_true_nt = nt_region(y_true,indexes,region)
  Precision = []
  Recall = []
  Sensitivity = []
  Specificity_1 = []
  producto_maximo=0
  th_vector = np.arange(0.7,1,0.01)
  for th in th_vector:
    Yhat_pred = NMS(y_hat, th, 0.1, indexes, region)
    Y_pred_nt = nt_region(Yhat_pred,indexes,region)
    TP = np.sum(Y_true_nt*Y_pred_nt)
    FP = np.sum(Y_pred_nt)-TP
    FN = np.sum(Y_true_nt)-TP
    TN = Y_true_nt.shape[0]*Y_true_nt.shape[2]-TP-FP-FN
    Precision.append(TP/(TP+FP))
    Recall.append(TP/(TP+FN))
    Sensitivity.append(TP/(TP+FN))
    Specificity_1.append(1-TN/(FP+TN))
    producto=Precision[-1]*Recall[-1]
    if producto>producto_maximo:
      th_max=th
      producto_maximo=producto
  plt.figure()
  plt.plot(Recall,Precision)
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.title('Precision - Recall')
  plt.savefig(f"{opcion}/P-R.png", format="png")

  plt.figure()
  plt.plot(Specificity_1,Sensitivity)
  plt.xlabel('1-Specificity')
  plt.ylabel('Sensitivity')
  plt.title('ROC curve')
  plt.savefig(f"{opcion}/ROC.png", format="png")
  return th_max


def YOLO_domain(optimizador=Adam,lr=0.001,momen=0,init_mode='glorot_normal',fun_act='linear',dp=0.2,regularizer=l2,w_reg=0,ventana=50000):
    tf.keras.backend.clear_session()
    w = 100
    n = 16
    inputs = tf.keras.Input(shape=(4,ventana, 1), name="input_1")
    LA = tf.keras.layers.Conv2D(n, (4, 50), strides=(1,1),activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(inputs)
    LA = tf.keras.layers.ZeroPadding2D(padding=((0,0), (0,49)))(LA)
    LA = tf.keras.layers.Conv2D(n, (1, 5), strides=(1,5),activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(LA)
    LA = tf.keras.layers.BatchNormalization()(LA)
    LA = tf.keras.layers.ReLU()(LA)
    LB = tf.keras.layers.Conv2D(n, (1, w), strides=(1,1),padding='same',activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(LA)
    LB = tf.keras.layers.BatchNormalization()(LB)
    LB = tf.keras.layers.ReLU()(LB)
    LB = tf.keras.layers.Conv2D(n, (1, w), strides=(1,1),padding='same',activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(LB)
    LB = tf.keras.layers.BatchNormalization()(LB)
    LB = tf.keras.layers.ReLU()(LB)
    LB = tf.keras.layers.Conv2D(n, (1, w), strides=(1,1),padding='same',activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(LB)
    LB = tf.keras.layers.BatchNormalization()(LB)
    LA = Add()([LB,LA])
    LA = tf.keras.layers.ReLU()(LA)

    w = 50
    LA = tf.keras.layers.Conv2D(n, (1, 2), strides=(1,2),activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(LA)
    LA = tf.keras.layers.BatchNormalization()(LA)
    LA = tf.keras.layers.ReLU()(LA)
    LB = tf.keras.layers.Conv2D(n, (1, w), strides=(1,1),padding='same',activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(LA)
    LB = tf.keras.layers.BatchNormalization()(LB)
    LB = tf.keras.layers.ReLU()(LB)
    LB = tf.keras.layers.Conv2D(n, (1, w), strides=(1,1),padding='same',activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(LB)
    LB = tf.keras.layers.BatchNormalization()(LB)
    LB = tf.keras.layers.ReLU()(LB)
    LB = tf.keras.layers.Conv2D(n, (1, w), strides=(1,1),padding='same',activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(LB)
    LB = tf.keras.layers.BatchNormalization()(LB)
    LA = Add()([LA,LB])
    LA = tf.keras.layers.ReLU()(LA)

    w = 10
    LA = tf.keras.layers.Conv2D(n, (1, 5), strides=(1,5),activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(LA)
    LA = tf.keras.layers.BatchNormalization()(LA)
    LA = tf.keras.layers.ReLU()(LA)
    LB = tf.keras.layers.Conv2D(n, (1, w), strides=(1,1),padding='same',activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(LA)
    LB = tf.keras.layers.BatchNormalization()(LB)
    LB = tf.keras.layers.ReLU()(LB)
    LB = tf.keras.layers.Conv2D(n, (1, w), strides=(1,1),padding='same',activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(LB)
    LB = tf.keras.layers.BatchNormalization()(LB)
    LB = tf.keras.layers.ReLU()(LB)
    LB = tf.keras.layers.Conv2D(n, (1, w), strides=(1,1),padding='same',activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(LB)
    LB = tf.keras.layers.BatchNormalization()(LB)
    LA = Add()([LA,LB])
    LA = tf.keras.layers.ReLU()(LA)

    w = 5
    LA = tf.keras.layers.Conv2D(n, (1, 2), strides=(1,2),activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(LA)
    LA = tf.keras.layers.BatchNormalization()(LA)
    LA = tf.keras.layers.ReLU()(LA)
    LB = tf.keras.layers.Conv2D(n, (1, w), strides=(1,1),padding='same',activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(LA)
    LB = tf.keras.layers.BatchNormalization()(LB)
    LB = tf.keras.layers.ReLU()(LB)
    LB = tf.keras.layers.Conv2D(n, (1, w), strides=(1,1),padding='same',activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(LB)
    LB = tf.keras.layers.BatchNormalization()(LB)
    LB = tf.keras.layers.ReLU()(LB)
    LB = tf.keras.layers.Conv2D(n, (1, w), strides=(1,1),padding='same',activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(LB)
    LB = tf.keras.layers.BatchNormalization()(LB)
    LA = Add()([LA,LB])
    LA = tf.keras.layers.ReLU()(LA)

    layers = tf.keras.layers.Conv2D(8, (1, 10), strides=(1,1),padding='same',activation='sigmoid', use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(LA)
    model = tf.keras.Model(inputs = inputs, outputs=layers)
    opt = optimizador(learning_rate=lr)
    model.compile(loss=loss_domains, optimizer=opt, metrics=[loss_precision_training])
    return model
