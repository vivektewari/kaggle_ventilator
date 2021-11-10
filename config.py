import time
from pathlib import Path
import random
import os
from data_loaders import *
from losses import *
from models import *
from param_options import *
from sklearn.preprocessing import RobustScaler
#from funcs import *
root ='/home/pooja/PycharmProjects/kaggle_ventilator/'
dataCreated = root+'/data/dataCreated/'
raw_data=root+ '/data/ventilator-pressure-prediction/'



#data_loader
data_loader_param =Data_load1_param
data_loader_param.file=raw_data+Data_load1_param.file

data_loader = SAKTDataset

#Model
model_param = Model1
model =GRUModel#fc_model_holder#

#loss function
loss_func = custom_L1Loss()


# metricSheetPath = root / 'metricSheet2.csv'
saveDirectory = root + '/outputs/weights/'
device = 'cpu'
config_id = str(os.getcwd()).split()[-1]
startTime = time.time()

lr = 0.0000000001

epoch = 10000

random.seed(23)




pre_trained_model ='/home/pooja/PycharmProjects/kaggle_ventilator/codes/rough/last.pth'
pre_trained_model ='/home/pooja/PycharmProjects/kaggle_ventilator/codes/fold0/checkpoints/last.pth'
#'/home/pooja/PycharmProjects/digitRecognizer/rough/localization/fold0/checkpoints/train.17.pth'


#pre_trained_model =None
