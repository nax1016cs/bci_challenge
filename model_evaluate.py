from EEGModels import ShallowConvNet
from EEGConvNets import SCCNet
from EEGModels import EEGNet
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from keras.optimizers import SGD
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn import metrics

import scipy
import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import keras.backend as K
import pandas as pd
import scipy.misc
import math
import copy
import statistics



# train_set = ["02", "06", "07", "11", "12", "13", "14", "16", "17", "18", "20", "21", "22", "23", "24", "26"]
train_set = ["02" ,"06", "07"]

test_set = ["01", "03", "04", "05", "08", "09", "10", "15", "19", "25"]
epoch = 500
kernel, ch , sample = 1, 56, 161

def square(x):
    return x * x

def safe_log(x):
    return K.log(x + 1e-7)

def log(x):
    return K.log(K.clip(x, min_value = 1e-7, max_value = 10000)) 

def load_data(sid, way):

    if way == "training":
        path = "bci_challenge_training_npz_/"
        npz = np.load(path + "Data_S" + str(sid) + "_Sess" +".npz")
        return npz["x_train"], npz["y_train"]

    elif way == "evaluate":
        path = "bci_challenge_testing_npz/"
        npz = np.load(path + "Data_S" + str(sid) + "_Sess" +".npz")
        return npz["x_train"]


def class_weight(y):
    weight = {}
    count_0 = 0
    count_1 = 0

    #  [1, 0] -> 0
    #  [0, 1] -> 1
    for num in y:
        if num[0] == [1]:
            count_0 += 1

        elif num[0] == [0]:
            count_1 += 1
    
    weight[0] = math.ceil(count_1/ min(count_0, count_1))
    weight[1] = math.ceil(count_0/min(count_0, count_1))
    # print("count: " , count_0, count_1)
    return weight


def create_model(network):
    if network == "EEGNet":
        model = EEGNet(nb_classes= 2, Chans = ch, Samples = sample, )
        
    elif network == 'ShallowNet':
        model = ShallowConvNet(nb_classes= 2, Chans = ch, Samples = sample,)
    
    elif network == 'SCCNet':
        model = SCCNet((kernel, ch, sample))

    return model

def create_checkpoint(network, train_method, sid, count):
    if network == "SCCNet":
        checkpoint = keras.callbacks.ModelCheckpoint('best_model/' + network + "_" + train_method + "_" + str(sid) + "_" + str(count) +'.h5',
                             monitor='loss', verbose=0, save_best_only=True, mode='min', period=1)
    else:
        checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model/' + network + "_" + train_method + "_" + str(sid) + "_" + str(count) +'.h5',
                             monitor='loss', verbose=0, save_best_only=True, mode='min', save_freq=1)
    
    return checkpoint

def load_best_model(network, train_method, sid, count ):
    if network == "SCCNet":
        model = load_model('best_model/' + network + "_" + train_method + "_" + str(sid) + "_" + str(count) +'.h5', 
                        custom_objects={'square':square, 'safe_log': safe_log, 'log':log})
    else:
        model = tf.keras.models.load_model('best_model/' + network + "_" + train_method + "_" + str(sid) + "_" + str(count) +'.h5',
                         custom_objects={'square':square, 'safe_log': safe_log, 'log':log})
    return model

def evaluate_result(model, x_test, y_test):
    ''' evaluate the model '''
    loss, acc = model.evaluate(x_test, y_test,  verbose=0)

    ''' confusion matrix '''
    predictions = model.predict(x_test) 
    cf_matrix = confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1))
    
    ''' auc value '''
    fpr, tpr, thresholds = metrics.roc_curve(y_test.argmax(axis=1)+1, predictions[:, 1], pos_label=2)
    auc = metrics.auc(fpr, tpr)
    print("acc:", acc)
    print("auc:", auc)
    print("confusion_matrix:\n", cf_matrix, "\n\n")


    return acc, auc, cf_matrix

def individual(network, sid, train_method, count):

    data_x, data_y = load_data(sid, "training")
    ratio = 4

    skf = KFold( n_splits = ratio)
    acc_list = []
    auc_list = []
    print("network:", network, "train_method:", train_method, "sid:", sid, "count:", count)

    for train_index, test_index in skf.split(data_x, data_y):
     
        # split the data into training data, validation data, and testing data
        x_train, x_test = data_x[train_index], data_x[test_index]
        y_train = tf.keras.utils.to_categorical(data_y[train_index], 2)
        y_test =  tf.keras.utils.to_categorical(data_y[test_index], 2)
        split_len = int(len(data_x)/ratio)
        x_val, y_val = x_train[-1 * split_len:], y_train[-1 * split_len:]
        x_train, y_train = x_train[:split_len*2], y_train[:split_len*2]
        
        # reshape the training data
        trial_num = 170 if train_method == "ind" else int (len(x_train) /ch /sample)
        x_train = np.reshape(x_train, (trial_num, kernel, ch, sample))
        weight = class_weight(y_train)

        # create the network
        model = create_model(network)
        # print(model.summary())

        # batch_size = int(len(data_x)/8) if train_method == "ind" else len(data_x)*2
        model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
        checkpoint = create_checkpoint(network, train_method, sid, count)
        # print(y_train.shape)
        history = model.fit(x_train, y_train , epochs = epoch, verbose = 0, validation_data = (x_val, y_val),
         class_weight = weight, shuffle = True, callbacks = [checkpoint]
        )

        model = load_best_model(network, train_method, sid, count)

        acc, auc, cf_matrix = evaluate_result(model, x_test, y_test)

        acc_list.append(acc)
        auc_list.append(auc)

    return (statistics.mean(acc_list), statistics.mean(auc_list))


def cross_subject(network, sid, train_method, count):
    print("network: ", network, "train_method: ", train_method, "sid: ", sid, "count: ", count)


    x_test, y_test = load_data(sid, "training")
    x_test, y_test = x_test[170:], y_test[170:]
    y_test = tf.keras.utils.to_categorical(y_test, 2)



    if train_method == "SI_FT":
        # load the SI model
        model = tf.keras.models.load_model('model/' + network + str(sid)+ '.h5',
                         custom_objects={'square':square, 'safe_log': safe_log, 'log':log})

        ''' reshape the training data '''
        x_train, y_train = load_data(sid, "training")
        x_train, y_train = x_train[:170], y_train[:170]
        y_train = tf.keras.utils.to_categorical(y_train, 2)

        ''' get class weight '''
        weight = class_weight(y_train)

        ''' reshape data '''
        trial_num =  int (len(x_train) /ch /sample)
        x_train = np.reshape(x_train, (170, kernel , ch, sample))


        checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model/' + network + "_" + train_method + "_" + str(sid) + "_" + str(count) +'.h5',
                                     monitor='loss', verbose=0, save_best_only=True, mode='min', save_freq=1)


        history = model.fit(x_train, y_train , epochs = epoch, verbose = 0, validation_split = 0.25, 
            class_weight = weight, shuffle = True, callbacks = [checkpoint]
        )

        model = tf.keras.models.load_model('best_model/' + network + "_" + train_method + "_" + str(sid) + "_" + str(count) +'.h5',
                         custom_objects={'square':square, 'safe_log': safe_log, 'log':log})

        acc, auc, cf_matrix = evaluate_result(model, x_test, y_test)
        return acc, auc

    elif train_method == "SI"  or "SD":
        first = True
        for i in train_set:
            if i != sid:
                if not first:
                    x_train = np.append(x_train, load_data(i, "training")[0])
                    y_train = np.append(y_train, load_data(i, "training")[1])

                else:
                    x_train, y_train = load_data(i, "training")
                    first = False


    if train_method == "SD":
        x_train = np.append(x_train, load_data(sid, "training")[0][:170])
        y_train = np.append(y_train, load_data(sid, "training")[1][:170])


    
    y_train = tf.keras.utils.to_categorical(y_train, 2)
    weight = class_weight(y_train)


    # reshape the training data
    trial_num =  int (len(x_train) /ch /sample)
    x_train = np.reshape(x_train, (trial_num, kernel, ch, sample))

    model = create_model(network)

    checkpoint = create_checkpoint(network, train_method, sid, count)
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
   
    history = model.fit(x_train, y_train , epochs = epoch, verbose = 0, validation_split = 0.25, 
        class_weight = weight, shuffle = True, callbacks = [checkpoint]
    )
    model = load_best_model(network, train_method, sid, count)

    acc, auc, cf_matrix = evaluate_result(model, x_test, y_test)

    if train_method == "SI":
        model.save('model/' + network + str(sid) + '.h5')

    return (acc, auc)


# def eval(network, sid, train_method, count):
#     print("====== " , sid ," =======")
#     x_test, y_test = load_data(sid, "training")
#     x_test, y_test = x_test[170:], y_test[170:]


#     y_test -= 1
#     y_test = np.array(y_test)
#     y_test = np.reshape(y_test, (len(y_test), 1))
#     y_test = tf.keras.utils.to_categorical(y_test, 2)

#     weight = class_weight(y_test)

#     return (0, 0)


def result(train_method):

    # 1 original 
    # 2 elu
    # 3 elu 
    # 4 elu
    # !!!!!!!!!! need to modify !!!!!!!!!!
    count = 999
    eeg = []
    shallow = []
    scc = []

    for i in train_set:
        if (train_method == "ind"):
            scc.append(individual("SCCNet", i, train_method, count))
            eeg.append(individual("EEGNet", i, train_method, count))
            shallow.append(individual("ShallowNet", i, train_method, count))
        else:
            scc.append(cross_subject("SCCNet", i, train_method, count))
            eeg.append(cross_subject("EEGNet", i, train_method, count))
            shallow.append(cross_subject("ShallowNet", i, train_method, count))
    
    print("====== " , train_method, " =============")
    print("eeg: ", eeg)
    print("shallow: ", shallow)
    print("scc: ", scc)

    # np.savez_compressed('test_npz/' + train_method + '_' + str(count)   + '.npz', eeg=eeg, shallow=shallow, scc=scc)


print("caculate independent............")
result("ind")
print("caculate subject independent............")
result("SI")
print("caculate subject independent and fine tuning............")
result("SI_FT")
print("caculate subject dependent............")
result("SD")

