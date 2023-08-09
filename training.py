import numpy as np
import random
import matplotlib.pyplot as plt
import gc
import sys
from auxiliary_functions.training_aux_fun import *
random.seed(10)

ventana = 50000
opcion = '/shared/home/sorozcoarias/coffea_genomes/Simon/YORO/data'
opcion_results = '/shared/home/sorozcoarias/coffea_genomes/Simon/YORO/results_training'

def remove_LTR_data(Y,c):
    Y[:,:,:,0] = Y[:,:,:,-1]*(-1)+Y[:,:,:,0]
    Y = Y[:,:,:,0:c-1]
    return Y


def load_set_z(partition,step,sample):
    Z = np.load(f'{opcion}/Z_{partition}{5000}.npy')
    for i in np.arange(2*step,sample+step,step):
        Z = np.concatenate((Z, np.load(f'{opcion}/Z_{partition}{i}.npy')))
    return Z


def load_set_x(partition,step,sample):
    X = np.load(f'{opcion}/X_{partition}{5000}.npy')[:,0:4,:]
    for i in np.arange(2*step,sample+step,step):
        X = np.concatenate((X, np.load(f'{opcion}/X_{partition}{i}.npy')[:,0:4,:]))
    return X


def load_set_y_without_LTR(partition,step,sample,a,b,c):
    Y = np.load(f'{opcion}/Y_{partition}{5000}.npy')[:,a:b,:,0:c]
    Y = remove_LTR_data(Y,c)
    for i in np.arange(2*step,sample+step,step):
        YY=np.load(f'{opcion}/Y_{partition}{i}.npy')[:,a:b,:,0:c]
        YY = remove_LTR_data(YY,c)
        Y = np.concatenate((Y, YY))
    return Y

def training_YORO(dataset):
    sample = 5000
    step = 5000
    n=step
    X_train = load_set_x('train',step,sample)
    X_dev = load_set_x('dev',step,sample)
    X_test = load_set_x('test',step,sample)
    if dataset == 'int_dom':
        a = 0
        b = 1
        c = 9
        Y_train = load_set_y_without_LTR('train',step,sample,a,b,c)
        Y_dev = load_set_y_without_LTR('dev',step,sample,a,b,c)
        Y_test = load_set_y_without_LTR('test',step,sample,a,b,c)
    else:
       return None
    Z_train = load_set_z('train',step,sample)
    Z_dev = load_set_z('dev',step,sample)
    Z_test = load_set_z('test',step,sample)
    
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
    filepath=f'{model_name}.hdf5'
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss_precision_training',save_weights_only=True,mode='max',save_best_only=True)
    history=model.fit(X_train, Y_train, epochs=30, callbacks=[checkpoint], batch_size=32, validation_data=(X_dev,Y_dev))

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['training','validation'])
    plt.savefig(f"{opcion_results}/loss.png", format="png")

    plt.figure()
    plt.plot(history.history['loss_precision_training'])
    plt.plot(history.history['val_loss_precision_training'])
    plt.xlabel('Epochs')
    plt.ylabel('Precision_training')
    plt.legend(['training','validation'])
    plt.savefig(f"{opcion_results}/monitor-PR.png", format="png")
    return [X_test, Y_test, Z_test]

def compute_metrics(X_test, Y_test, Z_test):
    model_name=f'{opcion_results}/AAYOLO_domain_V27'
    model = YOLO_domain()
    filepath='{}.hdf5'.format(model_name)
    model.load_weights(filepath)
    Yhat_test = model.predict(X_test)

    indexes=[3,3+5]
    region='dom'
    #region='internal'
    th_max=Precision_Recall_ROC_TE(Y_test,Yhat_test,indexes,region,opcion_results)
    print(th_max)
    sys.stdout.flush()

    threshold_NMS=0.1
    print('Las siguientes metrica corresponden a un threshold de 0.5')
    Yhat_pred = NMS(Yhat_test[:,:,:,0:9], 0.5, threshold_NMS,indexes)
    metricas_TE(Y_test, Yhat_pred, indexes,region)
    sys.stdout.flush()

    print('Las siguientes metrica corresponden a un threshold de 0.74')
    Yhat_pred = NMS(Yhat_test[:,:,:,0:9], 0.74, threshold_NMS,indexes)
    metricas_TE(Y_test, Yhat_pred, indexes,region)
    sys.stdout.flush()

    print('Las siguientes metrica corresponden a un threshold de 0.95')
    Yhat_pred = NMS(Yhat_test[:,:,:,0:9],0.95, threshold_NMS,indexes)
    metricas_TE(Y_test, Yhat_pred, indexes,region)
    sys.stdout.flush()

    print(f'Las siguientes metrica corresponden a th_max: {th_max}')
    threshold_presence=th_max
    threshold_NMS=0.1
    Yhat_pred = NMS(Yhat_test, threshold_presence, threshold_NMS,indexes)
    metricas_TE(Y_test, Yhat_pred, indexes,region)
    sys.stdout.flush()

    for index in [398,399,400,401,402]:
      Visualization_LTR(Yhat_pred[index,:,:,:],Y_test[index,:,:,:], indexes,region,opcion_results)
      print('Hay '+str(np.sum(Y_test[index,:,:,0]))+'dominios')
      sys.stdout.flush()

    malos = Plot_parity_TE(Y_test,Yhat_pred, indexes,region,opcion_results)
    print(malos)
    sys.stdout.flush()

def main():
    #X_test, Y_test, Z_test = training_YORO('int_dom')
    #gc.collect()
    #'''
    step = 5000
    sample = 5000
    a = 0
    b = 1
    c = 9
    X_test = load_set_x('test',step,sample)
    gc.collect()
    Y_test = load_set_y_without_LTR('test',step,sample,a,b,c)
    gc.collect()
    Z_test = load_set_z('test',step,sample)
    gc.collect()
    #'''
    compute_metrics(X_test, Y_test, Z_test)


if __name__ == "__main__":
    main()
