from catalyst.dl import SupervisedRunner, CallbackOrder, Callback, CheckpointCallback
import dicom2nifti

from sklearn.metrics import roc_curve, auc


from funcs import DataCreation,create_directories,lorenzCurve

import matplotlib.pyplot as plt


from config import *
from funcs import get_dict_from_class,count_parameters


from torch.utils.data import DataLoader
import pandas as pd

import torch




# inference pipe line
# 1.iterate over patients
# 2. for folder not with flir assign 0.5 prob and for foldrs with flair do following:
#     a. convert dcm to dicom2nifti
#     b. apply model
#     c . save predcition in dictionary format
#     d. post all iteration convert dictionary datafram-> to csv for submission
# e(optional) calculate auc and losses(for trianign data only
def inference(model_param,model_,data_loader_param,data_loader,pretrained=None):
    data_load = data_loader(**get_dict_from_class(data_loader_param))
    model = model_(**get_dict_from_class(model_param))
    count_parameters(model)
    if pretrained is not None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(pretrained, map_location=device)
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
        except:
            model.load_state_dict(checkpoint)
        model.eval()
    model.rc_seq = 0
    model.update_rc_seq()
    val_file =data_load.data


    loaders = {

        "valid": DataLoader(data_loader(data_frame=val_file, **get_dict_from_class(data_loader_param)),
                            batch_size=2048,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True,
                            drop_last=False)
    }

    runner = SupervisedRunner(
        model = model,
        output_key="logits",
        input_key="indep",
        target_key="targets")
    # scheduler=scheduler,
    predictions=[]

    for rc_seq in range(9):
        runner.model.rc_seq = rc_seq
        runner.model.update_rc_seq()
        loaders['valid'].dataset.rc_seq = rc_seq
        loaders['valid'].dataset.update_rc_seq()
        for prediction in runner.predict_loader(loader=loaders["valid"]):
            pred=prediction['logits'].detach().cpu().numpy().squeeze()
            predictions.extend(pred)
    return predictions


if 1:
    start_time=time.time()
    pre_trained_model = '/home/pooja/PycharmProjects/kaggle_ventilator/outputs/weights/gru_300.pth'
    pre_trained_model = '/home/pooja/PycharmProjects/kaggle_ventilator/codes/fold0/checkpoints/last.pth'
    submission=1
    if submission==1:
        file="test.csv"
    else :file = "train.csv"
    Data_load1_param.file = raw_data + file


    predictions=inference(model_param, model, Data_load1_param, data_loader, pretrained=pre_trained_model)
    predictions=torch.tensor(predictions)
    #df.to_csv(temp_path+"target.csv")
    dict_ = {}
    loop = 0
    for r in [5, 20, 50]:
        for c in [10, 20, 50]:
            dict_[str(r) + "_" + str(c)] = loop
            loop += 1
    df=pd.read_csv(raw_data+file)#,nrows=100000
    df=df.sort_values(['breath_id', 'time_step'], ascending=True).reset_index(drop=True)
    df['seq_RC'] = df.apply(lambda row: dict_[str(int(row['R'])) + "_" + str(int(row['C']))], axis=1)
    df.loc[df['u_out']==1,['seq_RC']]=10
    if submission==0:df['pressure2']=df['pressure']
    df['pressure']=0
    start=0

    for rc_seq in range(9):
        selection=df['seq_RC']==rc_seq
        temp=list(df[selection].groupby('breath_id').apply(lambda r: (len(r['u_in'].values))))#.reset_index()
        size = len(temp)
        mask = torch.zeros(size,40).bool()
        for bid in range(len(temp)):
            mask[bid][0:temp[bid]]=True
        pred=predictions[start:start+len(temp)][mask].squeeze()
        df.loc[selection,['pressure']]=pred.numpy()

        if submission==0:
            c = df[selection]
            print(abs(c[selection]['pressure'] - c[selection]['pressure2']).mean())
        start=start+size

    print(df.shape)

    if submission==0:
        print(abs(df[df.u_out==0]['pressure'] - df[df.u_out==0]['pressure2']).mean())
    else:
        df[['id', 'pressure']].to_csv(dataCreated + '/inference/result1.csv', index=False)
    print(time.time()-start_time)
    # lorenzCurve(df['actual'],df['pred'],save_loc=dataCreated+'/inference/com.png')
#6.9994540687462505,6.97762978852035,2.0+,0.97,1.571

