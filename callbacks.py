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
                 check_interval=1,
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

        self.visualizer = Visualizer()



    def on_epoch_end(self, state: Runner):
        """Event handler for epoch end.

        Args:
            runner ("IRunner"): IRunner instance.
        """
        # if (state.stage_epoch_step) % 2==0:
        #     self.get_grads_pic(state)
        #     get_layer_output()
        if self.directory is not None: torch.save(state.model.state_dict(), str(self.directory) + '/' +
                                                  self.model_name + "_" + str(
            state.stage_epoch_step) + ".pth")

        if (state.stage_epoch_step + 1) % self.check_interval == 0:

            preds =torch.where(state.batch['logits']>0.5,1,0)
            met=getMetrics(state.batch['targets'], preds)
            print("{} is {}".format(self.prefix,met ))
            self.visualizer.display_current_results(state.stage_epoch_step, state.epoch_metrics['train']['loss'],
                                                    name='train_loss')
            self.visualizer.display_current_results(state.stage_epoch_step, state.epoch_metrics['valid']['loss'],
                                                    name='valid_loss')
            self.visualizer.display_current_results(state.stage_epoch_step, state.epoch_metrics['train']['auc'],
                                                    name='train_auc')
            self.visualizer.display_current_results(state.stage_epoch_step, state.epoch_metrics['valid']['auc'],
                                                    name='valid _auc')
            self.visualizer.display_current_results(state.stage_epoch_step,
                                                    met[0], name='valid_accuracy')

    def on_batch_end(self, state):
        if state.loader_batch_step == 1  and state.is_train_loader:
            preds = torch.where(state.batch['logits'] > 0.5, 1, 0)
            met = getMetrics(state.batch['targets'], preds)
            self.visualizer.display_current_results(state.stage_epoch_step,
                                                    met[0], name='train_accuracy')


    def get_grads_pic(self, state,loc=""):
        # model train/valid step
        x, y = state.batch['image_pixels'],state.batch['targets']
        model=state.model
        y_hat =model(x)
        loss=state.criterion(y_hat,y)
        loss.backward()

        #get the blocks



        for block in state.model.state_dict().keys():
            fig = plt.figure()
            key = int(block.split('conv_blocks.')[1].split(".")[0])
            data=model.conv_blocks[key].conv.weight.grad.flatten(start_dim=1, end_dim=-1)

            min_max = [torch.min(data), torch.max(data)]
            plt.imshow(data, aspect='auto')
            plt.title(str(min_max[0]) + "_" + str(min_max[1]))
            fig.savefig(
               str(root)+'/diagnostics/'+'g_' + str(key) + "_grad_weights.png")
            plt.close()
            fig = plt.figure()
            data=model.conv_blocks[key].conv.weight.detach().clone().flatten(start_dim=1, end_dim=-1)

            min_max = [torch.min(data), torch.max(data)]
            plt.imshow(data, aspect='auto')
            plt.title(str(min_max[0]) + "_" + str(min_max[1]))
            fig.savefig(
                 str(root)+'/diagnostics/'+'w_'+ str(key) + "_weights.png")
            plt.close()
        plt.close('all')
class MetricsCallback_loc(Callback):

    def __init__(self,
                 directory=None,
                 model_name='',
                 check_interval=1,
                 input_key: str = "targets",
                 output_key: str = "logits",
                 prefix: str = "bound_loss,classification_loss,acc_pre_rec_f1",
                 func = getMetrics,
                 pixel =None,
                 image_scale=1):
        super().__init__(CallbackOrder.Metric)

        self.input_key = input_key
        self.output_key = output_key
        self.prefix = prefix
        self.directory = directory
        self.model_name = model_name
        self.check_interval = check_interval
        self.func = func
        self.drawing=DataCreation(image_path_=str(root) +'/images')
        self.vision_utils = vison_utils
        self.visualizer = Visualizer()
        self.pixel = pixel
        self.image_scale=image_scale

    # def on_batch_end(self,state: State):# #
    #     targ = state.input[self.input_key].detach().cpu().numpy()
    #     out = state.output[self.output_key]
    #
    #     clipwise_output = out[self.model_output_key].detach().cpu().numpy()
    #
    #     self.prediction.append(clipwise_output)
    #     self.target.append(targ)
    #
    #     y_pred = clipwise_output.argmax(axis=1)
    #     y_true = targ.argmax(axis=1)

    # score = f1_score(y_true, y_pred, average="macro")
    # state.batch_metrics[self.prefix] = score
    def draw_image(self, preds,color_intensity,msg= ""):
        i=0
        list_ = os.listdir(self.drawing.image_path)
        for img in list_:
            if img.find("pred") != -1:os.remove(self.drawing.image_path+"/"+img)
        list_ = os.listdir(self.drawing.image_path)
        list_.sort(key=lambda x: int(x.split("_")[0]))


        for img in list_[0:50]:
            self.drawing.draw_box(preds[i],color_intensity=color_intensity,scale=self.pixel,data=None,save_loc=self.drawing.image_path+"/"+img,msg =(msg[0][i], msg[1][i]),image_scale=None)
            i+=1
            if i==10 :break

    def rub_pred(self):
        i=0
        # list_=os.listdir(self.drawing.image_path)
        # list_.sort(key=lambda x: int(x.split("_")[0]))
        # for img in list_[0:9]:
        #     self.drawing.rub_box(data =None,dim=2,save_loc=self.drawing.image_path+"/"+img)
        #     i+=1
    def on_epoch_end(self, state: Runner):
        """Event handler for epoch end.

        Args:
            runner ("IRunner"): IRunner instance.
        """
        #print(torch.sum(state.model.fc1.weight),state.model.fc1.weight[5][300])
        #print(torch.sum(state.model.conv_blocks[0].conv1.weight))
        if (state.stage_epoch_step -1) % 50== 0:
            get_layer_output()
            self.get_grads_pic(state)
        if self.directory is not None: torch.save(state.model.state_dict(), str(self.directory) + '/' +
                                                  self.model_name + "_" + str(
            state.stage_epoch_step) + ".pth")

        if (state.stage_epoch_step + 1) % self.check_interval == 0:

            preds = state.batch['logits']
            box_count = int(preds.shape[1] / 15)
            #pred_class= torch.argmax(state.batch['logits'][:,:10], dim=1)
            temp = preds[:,:,:11]
            #pred_class=torch.argmax(temp.reshape((preds.shape[0],box_count*11)), dim=2)%11
            #accuracy_metrics=getMetrics(state.batch['targets'][:, 0], pred_class)
            loss=self.func(state.batch['targets'], preds)
            #print("{} is {}{}".format(self.prefix, loss,accuracy_metrics))
            self.visualizer.display_current_results(state.stage_epoch_step, state.epoch_metrics['train']['loss'],
                                                    name='train_loss')
            self.visualizer.display_current_results(state.stage_epoch_step, state.epoch_metrics['valid']['loss'],
                                                    name='valid_loss')
            # self.visualizer.display_current_results(state.stage_epoch_step, accuracy_metrics[0],
            #                                         name='accuracy')
            self.visualizer.display_current_results(state.stage_epoch_step, loss[0],
                                                    name='bounding_loss')
            self.visualizer.display_current_results(state.stage_epoch_step, loss[1],
                                                    name='classification_loss')
            self.visualizer.display_current_results(state.stage_epoch_step, loss[2],
                                                    name='background_loss')
    def on_epoch_start(self, state):


        c=0


    def on_batch_end(self,state):

        self.get_grads_pic(state)

        # if state.global_batch_step == 1:
        #     self.rub_pred()
        #torch.nn.utils.clip_grad_value_(state.model.parameters(), clip_value=1.0)
        target = state.batch['targets'].clone().detach()
        target[:, 1:] = target[:, 1:] / self.pixel
        if state.loader_batch_step == 1:
            if state.is_train_loader:
                model_output=state.get_model(1).model_outputs(state.batch['logits'])

                prec_rec = self.get_prec_recall(target,model_output)
                self.visualizer.display_current_results(state.stage_epoch_step, prec_rec[0],
                                                        name='precision')
                self.visualizer.display_current_results(state.stage_epoch_step, prec_rec[1],
                                                        name='recall')



        if state.loader_batch_step==1 and (state.global_epoch_step)%1 ==0 and state.is_train_loader:
            preds = state.batch['logits']
            # pred_class = torch.argmax(state.batch['logits'][:, :10], dim=1)
            #max_prob=torch.max(state.batch['logits'][:, :10], dim=1)[0]
            prioris,overlaps,prob,pred_locs=state.criterion.visualize_image(preds,state.batch['targets'])
            pred_locs=self.pred_boxes(model_output,[i for i in range(11)])
            #self.rub_pred()
            draw_list=[[prioris[i],pred_locs[i]] for i in range(len(prioris))]
            self.draw_image(draw_list, color_intensity=[(0,0,200),(200,200,0)],msg = (overlaps,prob))
            #self.draw_image(prioris, msg=(overlaps, prob))
            #self.get_grads(state)

            f=0
            #print("max_gradient is "+ torch.max(state.model.state_dict().values()[0]))
    def pred_boxes(self,model_pred_output,classes):
        n_class=len(classes)
        boxes = int(model_pred_output.shape[1] / (n_class + 4))
        scores = model_pred_output[:, :boxes * n_class]
        locs = model_pred_output[:, boxes * n_class:]
        pred_boxes = self.vision_utils.create_boxes(scores, n_class, locs)
        batch_size=model_pred_output.shape[0]
        return_boxes=[]
        for i in range(batch_size):
            pred_boxes_i = self.vision_utils.non_max_suppression(pred_boxes[i], classes, background_class=10,
                                                                 pred_thresh=0.7, overlap_thresh=0.5)
            return_boxes.append(pred_boxes_i)

        return return_boxes
    def get_prec_recall(self,targets,model_pred_output):
        """
        Algo:
        1.conver targets in batch,boxes,6
        2.conver preds in batch,boxes,6
        :param targets:
        :return:
        """
        batch_size=targets.shape[0]
        #1
        targets=torch.cat([targets[:,0].reshape((targets.shape[0],1)),torch.ones((targets.shape[0],1)),targets[:,1:].reshape((targets.shape[0],4))],dim=1)
        targets=targets.reshape((targets.shape[0],1,6))
        #2
        n_class = 11
        classes=[i for i in range(n_class)]

        boxes=int(model_pred_output.shape[1]/(n_class+4))

        pred_boxes=self.pred_boxes(model_pred_output,classes)
        for i in range(batch_size):
            true_boxes_i = targets[i]
            pred_boxes_i=pred_boxes[i]#self.vision_utils.non_max_suppression( pred_boxes[i], classes, background_class=11, pred_thresh=0.9, overlap_thresh=0.2)
            if i==0:
                tp_fp_np=self.vision_utils.tp_fp_fn(pred_boxes_i, true_boxes_i, classes[:-1], iou_threshold=0)
            else:
                tp_fp_np += self.vision_utils.tp_fp_fn(pred_boxes_i, true_boxes_i, classes[:-1], iou_threshold=0)
        tp_fp_np_all=torch.cat([torch.sum(tp_fp_np,dim=0).reshape((1,3)),tp_fp_np],dim=0)
        precision=tp_fp_np_all[:,0]/(tp_fp_np_all[:,0]+tp_fp_np_all[:,1])
        recall=tp_fp_np_all[:,0]/(tp_fp_np_all[:,0]+tp_fp_np_all[:,2])
        precision=torch.nan_to_num(precision,nan=0)
        recall= torch.nan_to_num(recall, nan=0)

        max_prec,max_rec,min_prec,min_rec=max(precision[1:]),max(recall[1:]),min(precision[1:]),min(recall[1:])
        return  [precision[0],recall[0],max_prec,max_rec,min_prec,min_rec]
    def get_grads(self, state):
        # model train/valid step
        x, y = state.batch['image_pixels'],state.batch['targets']
        model=state.model
        y_hat =model(x)
        loss=state.criterion(y_hat,y)
        loss.backward()

        #get the blocks
        if state.global_epoch_step==1:
            self.grad_column_names=['epoch']
            for block in state.model.state_dict().keys():
                self.grad_column_names.extend([block+"_"+i for i in  ['min','mean','max']])

            pd.DataFrame(columns=self.grad_column_names).to_csv(str(self.directory)+"//grads.csv")
        val_dict= {'epoch':state.global_epoch_step}
        funcs=[torch.min,torch.mean,torch.max]
        loop=1
        for block in state.model.state_dict().keys():
            key = int(block.split('conv_blocks.')[1].split(".")[0])
            temp=model.conv_blocks[key].conv.weight.grad
            for f in funcs:
                val_dict[self.grad_column_names[loop]]=[float(torch.abs(f(temp)))]
                loop+=1
        pd.DataFrame.from_dict(val_dict).to_csv(str(self.directory)+"//grads.csv",mode='a',header=False)
    def get_grads_pic(self, state,loc=""):
        # model train/valid step
        x, y = state.batch['image_pixels'],state.batch['targets']
        model=state.model
        y_hat =model(x)
        loss=state.criterion(y_hat,y)
        loss.backward()

        #get the blocks



        for block in state.model.state_dict().keys():
            fig = plt.figure()
            key = int(block.split('conv_blocks.')[1].split(".")[0])
            data=model.conv_blocks[key].conv.weight.grad.flatten(start_dim=1, end_dim=-1)

            min_max = [torch.min(data), torch.max(data)]
            plt.imshow(data, aspect='auto')
            plt.title(str(min_max[0]) + "_" + str(min_max[1]))
            fig.savefig(
               str(root)+'/diagnostics/'+'g_' + str(key) + "_grad_weights.png")
            plt.close()
            fig = plt.figure()
            data=model.conv_blocks[key].conv.weight.detach().clone().flatten(start_dim=1, end_dim=-1)

            min_max = [torch.min(data), torch.max(data)]
            plt.imshow(data, aspect='auto')
            plt.title(str(min_max[0]) + "_" + str(min_max[1]))
            fig.savefig(
                 str(root)+'/diagnostics/'+'w_'+ str(key) + "_weights.png")
            plt.close()
        plt.close('all')










