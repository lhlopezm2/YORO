import numpy as np
import random
import matplotlib.pyplot as plt
import gc
import sys
import os
from auxiliary_functions.training_aux_fun import *
random.seed(7)

ventana = 50000
opcion = '/shared/home/sorozcoarias/coffea_genomes/Simon/YORO/data'
opcion_results = '/shared/home/sorozcoarias/coffea_genomes/Simon/YORO/results_training'

def remove_LTR_data(Y,c):
    Y[:,:,:,0] = Y[:,:,:,-1]*(-1)+Y[:,:,:,0]
    Y = Y[:,:,:,0:c-1]
    return Y


def load_set_z(partition,step,sample, start):
    Z = np.load(f'{opcion}/Z_{partition}{start}.npy')
    for i in np.arange(start+step,sample+step,step):
        Z = np.concatenate((Z, np.load(f'{opcion}/Z_{partition}{i}.npy')))
    return Z


def load_set_x(partition,step,sample, start):
    X = np.load(f'{opcion}/X_{partition}{start}.npy')[:,0:4,:]
    for i in np.arange(start+step,sample+step,step):
        X = np.concatenate((X, np.load(f'{opcion}/X_{partition}{i}.npy')[:,0:4,:]))
    return X


def load_set_y(partition, step, sample, start):
    a = 1
    b = 2
    c = 5
    Y = np.load(f'{opcion}/Y_{partition}{start}.npy')[:,a:b,:,0:c]
    for i in np.arange(start+step,sample+step,step):
        YY=np.load(f'{opcion}/Y_{partition}{i}.npy')[:,a:b,:,0:c]
        Y = np.concatenate((Y, YY))
    return Y


def load_set_y_without_LTR(partition,step,sample, start):
    a = 0
    b = 1
    c = 9
    Y = np.load(f'{opcion}/Y_{partition}{start}.npy')[:,a:b,:,0:c]
    Y = remove_LTR_data(Y,c)
    for i in np.arange(start+step,sample+step,step):
        YY=np.load(f'{opcion}/Y_{partition}{i}.npy')[:,a:b,:,0:c]
        YY = remove_LTR_data(YY,c)
        Y = np.concatenate((Y, YY))
    return Y

def training_YORO(dataset, step, sample, start):
    X_train = load_set_x('train',step,sample, start)
    X_dev = load_set_x('dev',step,sample, start)
    X_test = load_set_x('test',step,sample, start)
    if dataset == 'int_dom':
        Y_train = load_set_y_without_LTR('train',step,sample, start)
        Y_dev = load_set_y_without_LTR('dev',step,sample, start)
        Y_test = load_set_y_without_LTR('test',step,sample, start)
    elif dataset == 'int_reg':
        Y_train = load_set_y('train',step,sample, start)
        Y_dev = load_set_y('dev',step,sample, start)
        Y_test = load_set_y('test',step,sample, start)
    else:
       return None
    Z_train = load_set_z('train',step,sample, start)
    Z_dev = load_set_z('dev',step,sample, start)
    Z_test = load_set_z('test',step,sample, start)
    
    sample = X_train.shape[0]+X_dev.shape[0]+X_test.shape[0]
    print(X_train.shape)
    print(Y_train.shape)
    print(X_dev.shape)
    print(Y_dev.shape)
    print(X_test.shape)
    print(Y_test.shape)
    print(sample)
    sys.stdout.flush()

    model_name=f'{opcion_results}/AAYOLO_domain_V27'
    model = YOLO_domain()
    #filepath=f'{model_name}.hdf5'
    filepath=f'{model_name}.h5'
    if os.path.exists(filepath):
        #model.load_weights(filepath)
        model = tf.keras.models.load_model(filepath)

    #checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss_precision_training',save_weights_only=True,mode='max',save_best_only=True)
    #checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss',save_weights_only=True,mode='min',save_best_only=True)
    history=model.fit(X_train, Y_train, epochs=20, batch_size=32, validation_data=(X_dev,Y_dev))
    model.save(filepath)
    plot_loss(history.history,opcion_results)
    return [X_test, Y_test, Z_test, history]


def plot_loss(history,opcion_results):
    plt.figure()
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['training','validation'])
    plt.savefig(f"{opcion_results}/loss.png", format="png")

    plt.figure()
    plt.plot(history['loss_precision_training'])
    plt.plot(history['val_loss_precision_training'])
    plt.xlabel('Epochs')
    plt.ylabel('Precision_training')
    plt.legend(['training','validation'])
    plt.savefig(f"{opcion_results}/monitor-PR.png", format="png")


def compute_metrics(X_test, Y_test, Z_test, section):
    model_name=f'{opcion_results}/AAYOLO_domain_V27'
    model = YOLO_domain()
    filepath='{}.hdf5'.format(model_name)
    model.load_weights(filepath)
    Yhat_test = model.predict(X_test)
    if section == 'int_dom':
        indexes = [3,3+5]
        region='dom'
    elif section == 'int_reg':
        indexes = [3,3+2]
        region='internal'
    try:
        th_max=Precision_Recall_ROC_TE(Y_test,Yhat_test,indexes,region,opcion_results)
    except:
        th_max=0.5
    print(th_max)
    sys.stdout.flush()

    threshold_NMS=0.1
    print('Las siguientes metrica corresponden a un threshold de 0.5')
    Yhat_pred = NMS(Yhat_test[:,:,:,0:9], 0.5, threshold_NMS,indexes,region)
    metricas_TE(Y_test, Yhat_pred, indexes,region)
    sys.stdout.flush()

    print('Las siguientes metrica corresponden a un threshold de 0.74')
    Yhat_pred = NMS(Yhat_test[:,:,:,0:9], 0.74, threshold_NMS,indexes,region)
    metricas_TE(Y_test, Yhat_pred, indexes,region)
    sys.stdout.flush()

    print('Las siguientes metrica corresponden a un threshold de 0.95')
    Yhat_pred = NMS(Yhat_test[:,:,:,0:9],0.95, threshold_NMS,indexes,region)
    metricas_TE(Y_test, Yhat_pred, indexes,region)
    sys.stdout.flush()

    print(f'Las siguientes metrica corresponden a th_max: {th_max}')
    threshold_presence=th_max
    threshold_NMS=0.1
    Yhat_pred = NMS(Yhat_test[:,:,:,0:9], threshold_presence, threshold_NMS,indexes,region)
    metricas_TE(Y_test, Yhat_pred, indexes,region)
    sys.stdout.flush()

    for i, index in enumerate([0,1,2,3,4]):
      Visualization_LTR(Yhat_pred[index,:,:,:],Y_test[index,:,:,:], indexes,region,opcion_results,i)
      print('Hay '+str(np.sum(Y_test[index,:,:,0]))+'dominios')
      sys.stdout.flush()

    malos = Plot_parity_TE(Y_test,Yhat_pred, indexes,region,opcion_results)
    print(malos)
    sys.stdout.flush()

def main():
    step = 5000
    sample = 55000
    section = 'int_dom'
    #section = 'int_reg'
    history_list=[]
    start = 55000
    """
    for i in range(start,sample+step,step):
        _, _, _, history = training_YORO(section, step, i, i)
        plt.close('all')
        gc.collect()
        history_list.append(history)
    """
    X_test = load_set_x('test',step,sample, start)
    gc.collect()
    Y_test = load_set_y_without_LTR('test',step,sample,start)
    gc.collect()
    Z_test = load_set_z('test',step,sample, start)
    gc.collect()
    """
    combined_history = {}
    for key in history_list[0].history:
        result = []
        for i in history_list:
            result = result +i.history[key]
        combined_history[key] = result
    plot_loss(combined_history,opcion_results)
    """
    compute_metrics(X_test, Y_test, Z_test, section)


if __name__ == "__main__":
    main()
