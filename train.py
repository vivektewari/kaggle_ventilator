import random

from data_loaders import *
from catalyst.dl import SupervisedRunner, CallbackOrder, Callback, CheckpointCallback
from config import *
from funcs import get_dict_from_class,count_parameters

import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from catalyst import dl
#from callbacks import MetricsCallback
from sklearn.model_selection import StratifiedKFold
import torch
def train(model_param,model_,data_loader_param,data_loader,loss_func,callbacks=None,pretrained=None):

    data_load = data_loader(**get_dict_from_class(data_loader_param))
    criterion = loss_func
    model = model_(**get_dict_from_class(model_param))
    count_parameters(model)
    # model = FCLayered(**get_dict_from_class(model_param,model))
    if pretrained is not None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(pretrained, map_location=device)
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
        except:
            model.load_state_dict(checkpoint)
        model.eval()
    optimizer = optim.SGD(model.parameters(), lr=lr)


    train = data_load.data
    groups=train.breath_id.unique()
    train_set=set(np.random.choice(list(groups),size=int(0.8*len(groups)),replace=False))
    valid_set=set(groups).difference(train_set)

    train_file=train[train.breath_id.isin(list(train_set))]
    val_file=train[train.breath_id.isin(list(valid_set))]





    print("train: {}, val: {}".format(train.breath_id.nunique(),val_file.breath_id.nunique()))

    loaders = {
        "train": DataLoader(data_loader( data_frame=train_file,**get_dict_from_class(data_loader_param)),
                            batch_size=2048,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True,
                            drop_last=False),
        "valid": DataLoader(data_loader(data_frame=val_file, **get_dict_from_class(data_loader_param)),
                            batch_size=2048,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True,
                            drop_last=False)
    }

    callbacks = callbacks
    runner = SupervisedRunner(

        output_key="logits",
        input_key="indep",
        target_key="targets")
    # scheduler=scheduler,

    runner.train(
        model=model,
        criterion=criterion,
        loaders=loaders,
        optimizer=optimizer,

        num_epochs=epoch,
        verbose=True,
        logdir=f"fold0",
        callbacks=callbacks,
    )

    # main_metric = "epoch_f1",
    # minimize_metric = False

if __name__ == "__main__":
    from callbacks import MetricsCallback

    callbacks = [MetricsCallback(input_key="targets", output_key="logits",
                         directory=saveDirectory, model_name='gru')]
    train(model_param,model,data_loader_param,data_loader,loss_func,callbacks,pretrained=pre_trained_model)