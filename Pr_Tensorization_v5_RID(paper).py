# -*- coding: utf-8 -*-
"""
Created on Wed May  1 10:28:19 2019

@author: Mohammad Eslami, Solale Tabarestani
"""

# Multilayer Perceptron
import pandas
import numpy
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
import tensorflow as tf



import tensorflow.keras
import math
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2DTranspose,Input, Reshape, Conv2D, Flatten
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import concatenate
import argparse
#from tensorflow.keras.utils.np_utils import to_categorical
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dropout
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr, spearmanr
from skimage import io
####################
####### Functions #############
####################

def custom_loss_2 (y_true, y_pred):
    A = tensorflow.keras.losses.mean_absolute_error(y_true, y_pred)
    #B = keras.losses.categorical_crossentropy(y_true[:,-4:], y_pred[:,-4:])
    return A





def lrelu(x): #from pix2pix code
    a=0.2
    # adding these together creates the leak part and linear part
    # then cancels them out by subtracting/adding an absolute value term
    # leak: a*x/2 - a*abs(x)/2
    # linear: x/2 + abs(x)/2

    # this block looks like it has 2 inputs on the graph unless we do this
    x = tf.identity(x)
    return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)

def lrelu_output_shape(input_shape):
    shape = list(input_shape)
    return tuple(shape)

from tensorflow.keras.layers import Lambda
layer_lrelu=Lambda(lrelu, output_shape=lrelu_output_shape, name='Leaky_ReLU')


def FnCreateTargetImages(Labels):
    OutputImages=np.zeros(shape=(len(Labels),23,23,3))    
    
    for i in range(len(Labels)):
        for j in range (4):
            if Labels.iloc[i,j]=='NL':
                OutputImages[i,:,j*5:(j+1)*5,0]=0
                OutputImages[i,:,j*5:(j+1)*5,1]=1
                OutputImages[i,:,j*5:(j+1)*5,2]=0
                
            elif Labels.iloc[i,j]=='MCI':
                OutputImages[i,:,j*5:(j+1)*5,0]=0
                OutputImages[i,:,j*5:(j+1)*5,1]=0
                OutputImages[i,:,j*5:(j+1)*5,2]=1
                
            elif Labels.iloc[i,j]=='MCI to Dementia':
                OutputImages[i,:,j*5:(j+1)*5,0]=1
                OutputImages[i,:,j*5:(j+1)*5,1]=0
                OutputImages[i,:,j*5:(j+1)*5,2]=1
                
            elif Labels.iloc[i,j]=='Dementia':
                OutputImages[i,:,j*5:(j+1)*5,0]=1
                OutputImages[i,:,j*5:(j+1)*5,1]=0
                OutputImages[i,:,j*5:(j+1)*5,2]=0
                
            # plt.figure()
            # plt.imshow(OutputImages[19,:,:,:])
    return OutputImages



def FnCreateValidLabes(Labels):
    return range(len(Labels))

     

####################
###### End of functions ##############    
####################    



n_splits=10

parser = argparse.ArgumentParser()
parser.add_argument("--output", )
parser.add_argument("--max_epochs",  )
parser.add_argument("--BatchSize", )
parser.add_argument("--k", )
parser.add_argument("--m", )
a = parser.parse_args()


a.max_epochs=4000
a.BatchSize=500
a.output='./v5-4000epochs/'
outputFolderChild=a.output+'/Images/'

import os
try:
    os.stat(a.output)
    os.stat(outputFolderChild)
except:
    os.mkdir(a.output) 
    os.mkdir(outputFolderChild)




####################
###### Reading Data ##############    
####################    

# AllDataset = pandas.read_csv('Data_XY_BLD_CutOff.csv', low_memory=False)
AllDataset = pandas.read_csv('Data_XY_BLD_v0.csv', low_memory=False)
AllDataset = AllDataset.set_index(AllDataset.RID)


AllDataset.columns

###################### MRI ######################
MRI_X = AllDataset.loc[:,['Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp', 'ICV']]
p=MRI_X.values

print(np.nanmin(MRI_X.ICV, axis=0))
print(np.nanmean(MRI_X.ICV, axis=0))
print(np.nanmax(MRI_X.ICV, axis=0))
MRI_X.ICV.isnull().sum()

MRI_Y = AllDataset.loc[:, ['DX_BLD', 'DX_6','DX_12', 'DX_24']]#
MRI_RID = AllDataset.RID
# normalize data
MRI_X = (MRI_X - MRI_X.mean())/ (MRI_X.max() - MRI_X.min())
#MRI_X=MRI_X+1
MRI_X =MRI_X.fillna(0)

###################### PET ######################
PET_X = AllDataset.loc[:,['FDG', 'PIB', 'AV45']]
PET_Y = AllDataset.loc[:, ['DX_BLD', 'DX_6','DX_12', 'DX_24']]#
PET_RID = AllDataset.RID

print(np.nanmin(PET_X.AV45, axis=0))
print(np.nanmean(PET_X.AV45, axis=0))
print(np.nanmax(PET_X.AV45, axis=0))
PET_X.AV45.isnull().sum()

# normalize data
PET_X = (PET_X - PET_X.mean()) / (PET_X.max() - PET_X.min())
#PET_X=PET_X+1
PET_X=PET_X.fillna(0)
###################### COG ######################
COG_X = AllDataset.loc[:, ['RAVLTimmediate', 'RAVLTlearning', 'RAVLTforgetting', 
                           'RAVLTpercforgetting','FAQ', 'MOCA',
                'EcogPtMem', 'EcogPtLang', 'EcogPtVisspat', 'EcogPtPlan',
                'EcogPtOrgan', 'EcogPtDivatt', 'EcogPtTotal',
                'EcogSPMem', 'EcogSPLang', 'EcogSPVisspat', 
                'EcogSPPlan', 'EcogSPOrgan', 'EcogSPDivatt', 'EcogSPTotal']]




COG_Y = AllDataset.loc[:, ['DX_BLD', 'DX_6','DX_12', 'DX_24']]#
COG_RID = AllDataset.RID


print(np.nanmin(COG_X.EcogSPTotal, axis=0))
print(np.nanmean(COG_X.EcogSPTotal, axis=0))
print(np.nanmax(COG_X.EcogSPTotal, axis=0))
COG_X.EcogSPTotal.isnull().sum()




# normalize data
COG_X = (COG_X - COG_X.mean()) / (COG_X.max() - COG_X.min())
#COG_X=COG_X+1
COG_X=COG_X.fillna(0)
###################### CSF ######################
CSF_X = AllDataset.loc[:,['ABETA', 'PTAU', 'TAU']]
CSF_Y = AllDataset.loc[:, ['DX_BLD', 'DX_6','DX_12', 'DX_24']]#
CSF_RID = AllDataset.RID

print(np.nanmin(CSF_X.ABETA, axis=0))
print(np.nanmean(CSF_X.ABETA, axis=0))
print(np.nanmax(CSF_X.ABETA, axis=0))
CSF_X.ABETA.isnull().sum()

# normalize data
CSF_X = (CSF_X - CSF_X.mean()) / (CSF_X.max() - CSF_X.min())
#CSF_X=CSF_X+1
CSF_X=CSF_X.fillna(0)
###################### Risk Factor ######################
RF_X_1 = AllDataset.loc[:,['AGE','PTEDUCAT']] #, 'APOE4', 'PTGENDER']]
RF_Y = AllDataset.loc[:, ['DX_BLD', 'DX_6','DX_12', 'DX_24']]#
RF_RID = AllDataset.RID

# normalize data
RF_X_1 = (RF_X_1 - RF_X_1.mean()) / (RF_X_1.max() - RF_X_1.min())
#RF_X_1=RF_X_1+1
RF_X_1=RF_X_1.fillna(0)

RF_X_A = AllDataset.loc[:,['APOE4']]# ,'PTEDUCAT']] 
print(np.nanmin(RF_X_A.APOE4, axis=0))
print(np.nanmean(RF_X_A.APOE4, axis=0))
print(np.nanmax(RF_X_A.APOE4, axis=0))
RF_X_A.APOE4.isnull().sum()
RF_X_A=RF_X_A-1
RF_X_A=RF_X_A.fillna(0)

RF_X_sex = AllDataset.loc[:,['PTGENDER']] #, 'APOE4', 'PTGENDER']]
RF_X_sex[RF_X_sex=='Male']=-1
RF_X_sex[RF_X_sex=='Female']=1
RF_X_sex=RF_X_sex.fillna(0)



import pandas as pd
RF_X = pd.concat([RF_X_1, RF_X_A, RF_X_sex], axis=1)#, RF_X_sex

##############################################


########################################
############## dataset cleaning #######
########################################

Labels_all_4class=COG_Y.iloc[:,-4:]

Labels_all_3class=Labels_all_4class.copy()
Labels_all_3class=Labels_all_3class.replace('MCI to Dementia','Dementia')

########################################
############## dataset balancing #######
########################################


Finall_Labels=Labels_all_3class

# Finall_Labels.to_csv('Final_Labels_3class.csv')



Counts_BLD=Finall_Labels['DX_BLD'].value_counts()
Counts_6=Finall_Labels['DX_6'].value_counts()
Counts_12=Finall_Labels['DX_12'].value_counts()
Counts_24=Finall_Labels['DX_24'].value_counts()

Balancing_value=np.max([Counts_BLD.MCI,Counts_6.Dementia,Counts_12.MCI,Counts_24.MCI])



########################################
############## genreating input outputs from data #######
########################################

TargetTensors=FnCreateTargetImages(Finall_Labels)
#plt.imshow(TargetTensors[11,:,:,:])
#print(Labels_all_4class.iloc[11])


X_all=[MRI_X.values, PET_X.values, COG_X.values, 
       CSF_X.values, RF_X.values]

YTrain = COG_Y
YTrain1 = YTrain.reset_index()
Y_all_tensors=TargetTensors
X_all_RIDs=RF_RID.values


#RF_X.head()





######################################################################################
################## NEtwork Architecture ##################################################
##########################################################################################

####################################### MRI FCN ###############################################
# mri FCN
MRI_inp_dim = MRI_X.shape[1]
MRI_visible = Input(shape=(MRI_inp_dim,), name='MRI')
hiddenMRI1 = Dense(2*MRI_inp_dim, kernel_initializer='normal', activation='linear')(MRI_visible)
hiddenMRI2 = hiddenMRI1
MRI_output = Dense(MRI_inp_dim, kernel_initializer='normal', activation='linear')(hiddenMRI2)

####################################### PET FCN ###############################################
PET_inp_dim = PET_X.shape[1]
PET_visible = Input(shape=(PET_inp_dim,), name='PET')
hiddenPET1 = Dense(2*PET_inp_dim, kernel_initializer='normal', activation='linear')(PET_visible)
hiddenPET2=hiddenPET1
PET_output = Dense(PET_inp_dim, kernel_initializer='normal', activation='linear')(hiddenPET2)

####################################### COG FCN ###############################################
# mri FCN
COG_inp_dim = COG_X.shape[1]
COG_visible = Input(shape=(COG_inp_dim,), name='COG')
hiddenCOG1 = Dense(2*COG_inp_dim, kernel_initializer='normal', activation='linear')(COG_visible)
hiddenCOG2=hiddenCOG1
COG_output = Dense(COG_inp_dim, kernel_initializer='normal', activation='linear')(hiddenCOG2)

####################################### CSF FCN ###############################################
CSF_inp_dim = CSF_X.shape[1]
CSF_visible = Input(shape=(CSF_inp_dim,), name='CSF')
hiddenCSF1 = Dense(2*CSF_inp_dim, kernel_initializer='normal', activation='linear')(CSF_visible)
hiddenCSF2=hiddenCSF1
CSF_output = Dense(CSF_inp_dim, kernel_initializer='normal', activation='linear')(hiddenCSF2)

####################################### RF FCN ###############################################
RF_inp_dim = RF_X.shape[1]
RF_visible = Input(shape=(RF_inp_dim,), name='RF')
hiddenRF1 = Dense(2*RF_inp_dim, kernel_initializer='normal', activation='linear')(RF_visible)
hiddenRF2 = hiddenRF1
RF_output = Dense(RF_inp_dim, kernel_initializer='normal', activation='linear')(hiddenRF2)

#################################### Concat FCN ###############################################

merge = concatenate([MRI_output, PET_output, COG_output, CSF_output,RF_output])
# interpretation layer
hidden1 = Dense(100, activation='linear')(merge)
hidden1 = Dropout(0.4)(hidden1)
hidden1_reshape = Reshape((10, 10, 1))(hidden1)
e_2=tensorflow.keras.layers.BatchNormalization()(hidden1_reshape)
e_2=layer_lrelu(e_2)

layer2D_1 = Conv2DTranspose(filters=10, kernel_size=(3,3), strides=(1, 1),padding="same", activation='linear')(e_2)
layer2D_1=tensorflow.keras.layers.BatchNormalization()(layer2D_1)
layer2D_1=layer_lrelu(layer2D_1)

layer2D_2 = Conv2DTranspose(filters=10, kernel_size=(3,3), strides=(1, 1), dilation_rate=(2,2),padding="same", activation='linear')(e_2)
layer2D_2=tensorflow.keras.layers.BatchNormalization()(layer2D_2)
layer2D_2=layer_lrelu(layer2D_2)


layer2D_3 = Conv2DTranspose(filters=10, kernel_size=(3,3), strides=(1, 1), dilation_rate=(3,3), padding="same", activation='linear')(e_2)
layer2D_3=tensorflow.keras.layers.BatchNormalization()(layer2D_3)
layer2D_3=layer_lrelu(layer2D_3)

############################################################################
layer2D_4 = concatenate([layer2D_1,layer2D_2,layer2D_3])
layer2D_4=tensorflow.keras.layers.BatchNormalization()(layer2D_4)
############################################################################

layer2D_5 = Conv2DTranspose(filters=100, kernel_size=(3,3), strides=(2, 2), kernel_regularizer=tensorflow.keras.regularizers.l2(0.01), activation='linear')(layer2D_4)
layer2D_5=tensorflow.keras.layers.BatchNormalization()(layer2D_5)
layer2D_5=layer_lrelu(layer2D_5)


layer2D_5 = Conv2DTranspose(filters=100, kernel_size=(3,3), strides=(2, 2), kernel_regularizer=tensorflow.keras.regularizers.l2(0.01), activation='linear')(layer2D_4)
layer2D_5=tensorflow.keras.layers.BatchNormalization()(layer2D_5)
layer2D_5=layer_lrelu(layer2D_5)


#layer2D_5_2 = Conv2DTranspose(filters=100, kernel_size=(3,3), strides=(2, 2), kernel_regularizer=tensorflow.keras.regularizers.l2(0.01), activation='linear')(layer2D_5)
#layer2D_5_2=tensorflow.keras.layers.BatchNormalization()(layer2D_5_2)
#layer2D_5_2=layer_lrelu(layer2D_5_2)



layer2D_6 = Conv2DTranspose(filters=3, kernel_size=(3,3), strides=(1, 1), kernel_regularizer=tensorflow.keras.regularizers.l2(0.08), activation='linear' )(layer2D_5)
layer2D_6=tensorflow.keras.layers.BatchNormalization()(layer2D_6)

output_1 = tensorflow.keras.layers.Activation('relu')(layer2D_6)#
model_tensorization=Model(inputs= [MRI_visible, PET_visible, COG_visible, CSF_visible, RF_visible], outputs=output_1)  



# model_tensorization.summary()
#dot_img_file = 'Network_23x23.png'
#tf.keras.utils.plot_model(model_tensorization, to_file=dot_img_file, show_shapes=True)



##############################################################################
################## End NEtwork Architecture ##########################################
######################################################################################


##########################################################################################
################## Training on folding ###################################################
##########################################################################################


# OPTIMIZER_1=tensorflow.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
OPTIMIZER_2=tensorflow.keras.optimizers.Adam(lr=0.001, beta_1=0.99, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# OPTIMIZER_3=tensorflow.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)


import time
model_tensorization.compile(loss=custom_loss_2, optimizer=OPTIMIZER_2)
model_tensorization.save_weights('SavedInitialWeights_tensors.h5')
callback_1 = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=500, restore_best_weights=True)
callback_2 = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=800, restore_best_weights=True)


for repeator in range(0,1):

    kfold = StratifiedKFold(n_splits, shuffle=True, random_state=repeator)
    FoldCounter=0
    for train, test in kfold.split(X_all[1], COG_Y.iloc[:,-1].values):
        FoldCounter=FoldCounter+1   
        
        model_tensorization.load_weights('SavedInitialWeights_tensors.h5')        
        Y_train_here_4Net=Y_all_tensors[train,:,:,:]
        X_train_here_4Net=[X_all[0][train], X_all[1][train], X_all[2][train], X_all[3][train],X_all[4][train]]
        X_train_here_names=X_all_RIDs[train]
        
        print('---Repeat No:  ', repeator+1, '  ---Fold No:  ', FoldCounter)        
        
        start_time = time.time()
        History = model_tensorization.fit(X_train_here_4Net, Y_train_here_4Net, 
        validation_split=0.1,  epochs=a.max_epochs, batch_size=a.BatchSize,
         verbose=1 , callbacks=[callback_1, callback_2]) # , callbacks=[callback_1, callback_2]
        elapsed_time = time.time() - start_time
        print('----- train elapsed time:', elapsed_time)


        # summarize history for loss
        plt.plot(History.history['loss'])
        plt.plot(History.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.grid()
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(a.output+'Fold_'+str(FoldCounter)+'_History.png')
        plt.close()

        X_test_here_4Net=[X_all[0][test], X_all[1][test], X_all[2][test], X_all[3][test], X_all[4][test]]
        Y_test_here_4Net=Y_all_tensors[test,:,:,:]
        X_test_here_names=X_all_RIDs[test]
        
        for i in range(len(test)):            
            plt.figure()
            plt.subplot(122)
            
            start_time = time.time()
            Output=model_tensorization.predict(X_test_here_4Net)[i,:,:,:]
            elapsed_time = time.time() - start_time
            print('----- test elapsed time for one subject:', elapsed_time)
            plt.imshow(((Output)))

            plt.subplot(121)
            plt.imshow(Y_test_here_4Net[i,:,:,:])
            plt.savefig(a.output+'Fold_'+str(FoldCounter)+ '_Test_RID_'+str(X_test_here_names[i]), dpi=300)
            plt.close()
            
            io.imsave(outputFolderChild+'Fold_'+str(FoldCounter)+ '_Test_RID_'+str(X_test_here_names[i])+'_output.png',model_tensorization.predict(X_test_here_4Net)[i,:,:,:])
            io.imsave(outputFolderChild+'Fold_'+str(FoldCounter)+ '_Test_RID_'+str(X_test_here_names[i])+'_target.png',Y_test_here_4Net[i,:,:,:])
            
            
        for i in range(0,100,2):
            plt.figure()
            plt.subplot(122)
        
            
            plt.imshow(((model_tensorization.predict(X_train_here_4Net)[i,:,:,:])))

            
            plt.subplot(121)
            plt.imshow(Y_train_here_4Net[i,:,:,:])
            plt.savefig(a.output+'Fold_'+str(FoldCounter)+'_Trained_RID_'+str(X_train_here_names[i]), dpi=300)
            plt.close()
        
            io.imsave(outputFolderChild+'Fold_'+str(FoldCounter)+ '_Train_RID_'+str(X_train_here_names[i])+'_output.png',model_tensorization.predict(X_train_here_4Net)[i,:,:,:])
            io.imsave(outputFolderChild+'Fold_'+str(FoldCounter)+ '_Train_RID_'+str(X_train_here_names[i])+'_target.png',Y_train_here_4Net[i,:,:,:])

        
            
