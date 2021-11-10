from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import RobustScaler
import torch,itertools,cv2,time,random
import pydicom as di
import torchio as tio

import numpy as np
import os
maxrows =50000

class SAKTDataset(Dataset): # borrowed from riid challange work
    def __init__(self,file, data_frame=None, max_seq=40,rc_seq=0):  # HDKIM 100
        super(SAKTDataset, self).__init__()
        self.max_seq = max_seq
        if data_frame is None:df = pd.read_csv(file)#,nrows=100000
        else :df=data_frame
        self.data=df
        self.dict_RC = self.seq_RC()
        self.df=self.preprocess(df)
        self.rc_seq=rc_seq
        self.update_rc_seq()



    def update_rc_seq(self):
        df = self.df[self.df['seq_RC']==self.rc_seq]
        self.get_group(df)
    def __len__(self):
        return len(self.user_ids)
    def seq_RC(self):
        dict_={}
        loop=0
        for r in [5,20,50]:
            for c in [10,20,50]:
                dict_[str(r)+"_"+str(c)]=loop
                loop+=1
        return dict_

    def preprocess(self,df):
        if 'pressure' not in df.columns: df['pressure']=0

        df = df.sort_values(['breath_id', 'time_step'], ascending=True).reset_index(drop=True)
        c=df.groupby(['breath_id'])['time_step'].shift(-1).fillna(0)
        d=df.groupby(['breath_id'])['time_step'].shift(0).fillna(0)
        df['time_step_delta'] = c-d#df.groupby('breath_id')['time_step'].shift(-1).fillna(0)-df.groupby('breath_id')['time_step']#(df['u_in']).groupby(df['breath_id']).cumsum()

        df['u_in']=df.apply(lambda x: x['u_in']*x['time_step_delta'],axis=1)
        df = df[df.u_out == 0]
        #df = df[df.R == 5][df.C == 20]
        df['seq_RC']=df.apply(lambda row:self.dict_RC[str(int(row['R']))+"_"+str(int(row['C']))],axis=1)

        # scaler = RobustScaler()
        # df.u_in = scaler.fit_transform(np.array(df.u_in).reshape(-1,1))
        # df.pressure= scaler.fit_transform(np.array(df.pressure).reshape(-1, 1))

        # df.pressure = df.pressure/100 #-np.mean(df.pressure ))/ np.std(df.pressure )
        # df.u_in = df.u_in/100 # -np.mean(df.u_in)) / np.std(df.u_in)
        #
        # df.time_step=df.time_step/4
        return df.fillna(0)
    def get_group(self,df):
        group =df[['breath_id', 'u_in', 'pressure', 'time_step','time_step_delta']].groupby('breath_id').apply(lambda r: (
        r['u_in'].values,
        r['pressure'].values,
        r['time_step'].values,
        r['time_step_delta'].values))
        self.samples = group
        self.user_ids = [x for x in group.index]


    def __getitem__(self, index):
        user_id = self.user_ids[index]
        q_, qa_, ts_ ,ucs_= self.samples[user_id]
        seq_len = len(q_)
        mask=torch.zeros(self.max_seq).bool()
        mask[:seq_len]=True
        q = torch.zeros(self.max_seq)
        qa = torch.zeros(self.max_seq)
        ts = torch.zeros(self.max_seq)
        ucs = torch.zeros(self.max_seq)
        shp=torch.tensor(seq_len,dtype=torch.int)
        q[0:seq_len]= torch.tensor(q_[0:seq_len])
        qa[0:seq_len]= torch.tensor(qa_[0:seq_len])
        ts[0:seq_len] = torch.tensor(ts_[0:seq_len])
        ucs[0:seq_len] = torch.tensor(ucs_[0:seq_len])
        indep = torch.unsqueeze(q,1)#q#torch.cat((torch.unsqueeze(q,1), torch.unsqueeze(ts,1),torch.unsqueeze(ucs,1)), dim=1)
        # target_id = q[1:]
        # label = qa[1:]

        #         x = np.zeros(self.max_seq-1, dtype=int)
        #         x = q[:-1].copy()
        #         x += (qa[:-1] == 1) * self.n_skill

        return {'indep':(ucs[0],indep),'targets':(qa,mask)}




if __name__ == "__main__":pass
    # from funcs import get_dict_from_class
