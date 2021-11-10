import pandas as pd
import torch
import numpy as np
from funcs import getMetrics,DataCreation,vison_utils
from utils.visualizer import Visualizer
import os,cv2
# from diag2 import get_layer_output
from catalyst.dl  import  Callback, CallbackOrder,Runner
import matplotlib.pyplot as plt
from config import root

class MetricsCallback(Callback):

    def __init__(self,
                 directory=None,
                 model_name='',
                 check_interval=10,
                 input_key: str = "targets",
                 output_key: str = "indep",
                 prefix: str = "l1loss",

                 ):
        super().__init__(CallbackOrder.Metric)

        self.input_key = input_key
        self.output_key = output_key
        self.prefix = prefix
        self.directory = directory
        self.model_name = model_name
        self.check_interval = check_interval
        self.rc_seq=0
        self.all_loop=0
        self.loop=0
        self.visualizer = Visualizer()
        self.loss_dict={}
        self.count_looper={}
        for i in range(9):
            self.loss_dict[i]=1
            self.count_looper[i]=0


    def on_epoch_start(self, state: Runner):
        if (state.stage_epoch_step -1) % 100 == 0: self.change_rc_seq(state)
    def on_epoch_end(self, state: Runner):

        """Event handler for epoch end.

        Args:
            runner ("IRunner"): IRunner instance.
        """

        self.count_looper[self.rc_seq] += 1


        if (state.stage_epoch_step+1) % self.check_interval == 0:
                self.visualizer.display_current_results(self.count_looper[self.rc_seq], state.epoch_metrics['valid']['loss'],
                                                        name='valid_loss_'+str(self.rc_seq))
                self.loss_dict[self.rc_seq]=state.epoch_metrics['valid']['loss']**2


    def select_rc_seq(self):
        vals = self.loss_dict.values()
        total = sum(vals, 0.0)
        prob = np.array(list(vals))/total
        draw = np.random.choice(list(self.loss_dict.keys()), 1,
                      p=prob)
        self.rc_seq = draw[0]




    def iter_choice(self):
        if self.rc_seq == 8:
            self.rc_seq = 0
            self.all_loop += self.loop
        else:
            self.rc_seq = self.rc_seq + 1

        train_set = [6,8]  # [i for i in range(9)]#[6,8]#
        if self.rc_seq not in train_set: self.iter_choice()

    def change_rc_seq(self,state):

        if self.directory is not None : torch.save(state.model.state_dict(),
                                                                           str(self.directory) + '/' +
                                                                         self.model_name + "_" + str(
                                                                                self.all_loop ) + ".pth")
        #self.count_looper[self.rc_seq] = self.loop
        #self.select_rc_seq()
        self.iter_choice()
        if self.loss_dict[self.rc_seq]>1.2:
            state.optimizer.param_groups[0]['lr']=1
        elif self.loss_dict[self.rc_seq]>0.5:
            state.optimizer.param_groups[0]['lr'] = 0.1
        else:
            state.optimizer.param_groups[0]['lr'] = 0.01



        #self.loop = self.count_looper[self.rc_seq]

        state.model.rc_seq=self.rc_seq
        state.model.update_rc_seq()
        state.loaders['train'].dataset.rc_seq=self.rc_seq
        state.loaders['valid'].dataset.rc_seq=self.rc_seq
        state.loaders['train'].dataset.update_rc_seq()
        state.loaders['valid'].dataset.update_rc_seq()




