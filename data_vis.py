import pandas as pd
import seaborn as sns
import enum
import os,glob
from unet import UNet
from scipy import stats

from config import root,dataCreated
import matplotlib.pyplot as plt
from commonFuncs import packing
from pathlib import Path
import torch,cv2

import numpy as np
import random
import torchio as tio
from tqdm import tqdm
import multiprocessing
import torch.nn.functional as F
if 0:
    # finding what will be best input 4 channel,9, 12 channel?


    import seaborn as sns
    df0 = pd.read_csv(Path(dataCreated) / 'image_info' / 'images2.csv', dtype='str')
    c=df0.describe()
    pid='patient_id'
    df0['SliceThickness']=df0['SliceThickness'].astype('float32').astype('int')
    data =df0.drop_duplicates(subset=[pid,'test_type','plane'])
    for pl in ['test_type','plane','SamplesPerPixel','PhotometricInterpretation','SliceThickness']:
        data2 = data.groupby(pl)["image_name"].count()
        data2.plot.pie(autopct="%.1f%%")
        plt.savefig(str(root)+"data/diagnostics/data_plots/"+pl+".png")
        plt.close()
        data2.plot.bar()
        plt.savefig(str(root) + "data/diagnostics/data_plots/" + pl + "_account_count.png")
        plt.close()
    for heat_maps in [['test_type','plane'],['plane','SliceThickness'],['test_type','SliceThickness']]:
        data = df0[heat_maps+[pid]].drop_duplicates(subset=heat_maps+[pid]).groupby(heat_maps).size().unstack(fill_value=0)
        #plt.imshow(data, cmap='hot', interpolation='nearest')
        sns.heatmap(data, annot=True, cmap='Blues', fmt='g')
        plt.savefig(str(root) + "data/diagnostics/data_plots/" + "".join(heat_maps) +"_heat.png")
        plt.close()
    # plt.hist(df0[['area']],bins=10, alpha=0.5,histtype='stepfilled' )
    # plt.savefig(str(root) + "data/diagnostics/data_plots/" + "area.png")
    # plt.close()
if False:
    #analysing best crop for images
    df0 = pd.read_csv(dataCreated / 'image_info' / 'images1.csv', dtype='str')
    temp = df0['coords'].apply(lambda x: packing.unpack(x))
    coords = torch.tensor(list(temp))
    df0['x_len'] = coords[:, 2] - coords[:, 0]
    df0['y_len']=coords[:, 3] - coords[:, 1]
    plt.hist(df0[['x_len','y_len']], bins=10,density=True , histtype='bar',cumulative=1)


    plt.savefig(str(root) + "/diagnostics/data_plots/" + "len.png")

if 0:
    df0 = pd.read_csv(dataCreated / 'image_info' / 'images4.csv')
    image_vars = ['image_shape_x', 'image_shape_y', 'pixel_mean', 'pixel_std', 'pixel_max', 'pixel_min', 'pixel_0.75',
                  'Pixel_0.9']

    for var in image_vars:
        temp=df0[df0[var]!=0][[var]]
        plt.hist(temp, bins=100,density=True , histtype='bar',cumulative=1)
        plt.savefig(str(root) + "/diagnostics/data_plots/" + var+"_hist.png")
        plt.close()
        plt.hist(temp, bins=100, density=False, histtype='bar', cumulative=0)
        plt.savefig(str(root) + "/diagnostics/data_plots/" + var + "_hist2.png")
        plt.close()
if False: # checking in slice location and occupied perc distribution
    var='slice_loc_perc_'
    df0 = pd.read_csv(dataCreated + '/image_info/images2.csv')
    df1 = pd.read_csv(dataCreated+ '/image_info/images6.csv')
    df1=df1[df1['image_shape_x']==512]
    df1 = df1[df1['image_shape_y'] == 512]

    df2=pd.merge(df1 ,df0[['patient_id','test_type','image_name','SliceLocation']],on =['patient_id','test_type','image_name'])
    df2=df2[df2['SliceLocation']!='error']
    df2['SliceLocation']= df2['SliceLocation'].astype('float64')
    for plane in ['coronal','axial','sagittal']:
        sns.scatterplot(data=df2[df2['plane']==plane][["SliceLocation","occupied_perc","plane"]].fillna(0), x="SliceLocation", y="occupied_perc", hue="plane")
        plt.savefig(str(root) + "/data/diagnostics/data_plots/" + var +plane+ "_scatter.png")
        plt.close()
if 0:#checking slice location avaiblty for each value
    var='slice_loc'
    df0 = pd.read_csv(dataCreated + '/image_info/images2.csv')
    df0 = df0[df0['SliceLocation'] != 'error']
    df0=df0[df0['test_type']=='T1w']
    df0 = df0[df0['plane'] == 'axial']
    df0['SliceLocation_integer'] = df0['SliceLocation'].astype('float32').apply(lambda x:round(x))
    df0['SliceLocation_integer'] = df0['SliceLocation_integer'].apply(lambda x:x if x%5==0 else x-1 if (x-1)%5==0 else x+1 if (x+1)%5==0 else -1)
    df0=df0[df0['SliceLocation_integer']!=-1]
    #df0=df0.drop_duplicates(['patient_id','test_type','SliceLocation_integer'])
    df1=df0.groupby(['SliceLocation_integer'])['patient_id'].count()
    plt.hist(df0[['SliceLocation_integer']], density=False,bins=400, histtype='step', cumulative=0)

    plt.savefig(str(root) + "/data/diagnostics/data_plots/" + var +"_hist.png")
    plt.close()
if 0:
    # testing for slice location offset. Compute difference of first and last visible slides

    included_plain = ['axial']
    df0 = pd.read_csv(dataCreated + '/image_info/images2.csv')
    df1 = pd.read_csv(dataCreated + '/image_info/images5.csv')
    print('start_'+str(df0.shape[0]))
    df0 = df0[df0['SliceLocation'] != 'error']
    print('waterfall_slice_loc_error'+str(df0.shape[0]))

    df0['SliceLocation'] = df0['SliceLocation'].astype('float32')  # .apply(lambda x: round(x))
    # df0 = df0.drop_duplicates(['patient_id', 'test_type', 'SliceLocation_integer'])
    print('df1_start' + str(df1.shape[0]))
    df2 = pd.merge(df1.drop('plane',axis=1), df0[['patient_id', 'test_type', 'image_name', 'SliceLocation','plane']],
                   on=['patient_id', 'test_type', 'image_name'], how='inner')
    print('inner_join loss' + str(df2.shape[0]))
    df2 = df2[df2['plane'].isin(included_plain)]
    print('axial complement loss' + str(df2.shape[0]))

    df2 = df2.sort_values(['patient_id', 'test_type', 'SliceLocation'])
    df2['pixel_max'] = df2['pixel_max'].apply(lambda x: int(x > 0))
    df2 = df2.groupby(['patient_id', 'plane', 'test_type'])[
        'image_name', 'pixel_max', 'SliceLocation'].agg(list).reset_index().rename(columns={'image_name': 'images'})
    df2['pixel_max_str'] = df2['pixel_max'].apply(lambda x: "".join(map(str, x)))
    df2['pixel_max_val_pass'] = df2['pixel_max_str'].apply(lambda x: [len(x.split('01')[0])+1,
                                                           len(x)-len(x.split('10')[1])-2] if len(x.split('01')) == 2 and len(
                                                               x.split('10')) == 2 else -1)
    print('post_listing' + str(df2.shape[0]))
    df2=df2[ df2['pixel_max_val_pass']!=-1]
    print('slice_location _validation_loss_' + str(df2.shape[0]))
    df2['slice_diff'] = df2.apply(lambda row: row['SliceLocation'][row['pixel_max_val_pass'][1]] -
                                              row['SliceLocation'][row['pixel_max_val_pass'][0]], axis=1)
    plt.hist(df2[['slice_diff']] , density=False, bins=400, histtype='step', cumulative=0)

    plt.savefig(str(root) + "/data/diagnostics/data_plots/" + 'slice_diff' + "_hist.png")
    plt.close()
    seed=23
    random.seed(seed)
    torch.manual_seed(seed)
    num_workers = multiprocessing.cpu_count()
    plt.rcParams['figure.figsize'] = 12, 6
    images_dir = Path(dataCreated + '/preprocessed2/')
    image_paths = sorted(images_dir.glob('*/T1w/*.png'))
if 0: #pixel hist plotting


    def plot_histogram(axis, tensor, num_positions=100, label=None, alpha=0.05, color=None):
        values = tensor.numpy().ravel()
        if values.max()<0.1:return None
        values=values[values!=0]
        if np.std(values)<0.001:return None
        kernel = stats.gaussian_kde(values)

        positions = np.linspace(values.min(), values.max(), num=num_positions)
        histogram = kernel(positions)
        kwargs = dict(linewidth=1, color='black' if color is None else color, alpha=alpha)
        if label is not None:
            kwargs['label'] = label
        axis.plot(positions, histogram, **kwargs)






    test_type='T1w'
    images_dir = Path(str(dataCreated)+"/nii/")
    image_paths = sorted(images_dir.glob('*/'+test_type+'/resampled_eq.nii'))

    fig, ax = plt.subplots(dpi=100)
    color='cyan'

    for path in tqdm(image_paths[0:50]):
        tensor = tio.ScalarImage(path).data

        if 'FLAIR' in str(path):
            color = 'red'
        elif 'T1wCE' in str(path):
            color = 'green'
        elif 'T1w' in str(path):
            color = 'blue'
        elif 'T2w' in str(path):
            color = 'black'
        plot_histogram(ax, tensor, color=color)
    # ax.set_xlim(0, 50000)
    # ax.set_ylim(0, 0.0005);
    ax.set_title('Original histograms of all samples')
    ax.set_xlabel('Intensity')
    ax.grid()
    #plt.savefig(str(root) + "/data/diagnostics/data_plots/" + 'pixels_dist_trans' + "_nii_hist.png")
    plt.savefig(str(root) + "/data/diagnostics/data_plots/" + 'pixels_dist_trans' + "eq_nii_hist.png")

if 1: #plotting slices for each nii to see the diferences
    def preprocess(input_):

        #below line is not required for nii file model building
        #input_=input_.reshape((input_.shape[0],1,)+tuple(input_.shape[1:]))*1.0
        mean=torch.mean(torch.where(input_>0.0,input_,torch.tensor(np.nan)),dim=(0,1,2),keepdim=True)#.reshape((input_.shape[0:3]+(1,1))).expand(input_.shape)
        mean=torch.nan_to_num(mean,nan=0)
        std = torch.std(torch.where(input_>0.0,input_,mean),dim=(0,1,2),keepdim=True)
        std=torch.where(std==0.0,torch.tensor(0.0001),std)
        std = torch.nan_to_num(std, nan=0)

        x=(input_-mean)/std

        return x
    def plot_slice_from_3d(nii_file,slice_number,output_loc):
        mas = tio.ScalarImage(nii_file)
        splits=str(nii_file).split("/")

        data = np.array(mas.data)[0]
        p = np.argmax(np.sum(data, axis=(0, 1)))
        data =preprocess(torch.tensor(data)*1.0).numpy()
        #data=np.flip(data,axis=1)
        #data = np.flip(data, axis=1)

        for slice in slice_number:
            im=data[ :, slice,:].reshape((240,155))*5000
            name = splits[-3] + splits[-2] +str(slice)+ ".png"
            cv2.imwrite(output_loc+name,im.astype(np.uint16))


    images_dir=Path('/home/pooja/PycharmProjects/rsna_cnn_classification/data/dataCreated/rough/')
    output_loc='/home/pooja/PycharmProjects/rsna_cnn_classification/rough/data_vis_slices/'
    image_paths = sorted(images_dir.glob('adjusted.nii'))
    #output_loc = '/home/pooja/PycharmProjects/rsna_cnn_classification/rough/data_vis_slices_2/'
    #image_paths = sorted(images_dir.glob('00018/FLAIR/task.nii'))
    for p in image_paths[0:10]:
        plot_slice_from_3d(p, [5*i for i in range(3,25)], output_loc)

if 0:
    images_dir=Path('/home/pooja/PycharmProjects/rsna_cnn_classification/data/dataCreated/ni_copy/nii/')
    image_paths = sorted(images_dir.glob('*/FLAIR/tumor.nii'))
    output_loc='/home/pooja/PycharmProjects/rsna_cnn_classification/rough/augmentation/'
    flip = tio.RandomFlip(axes=('LR',))
    rb=tio.RandomBiasField()
    re=tio.RandomElasticDeformation()
    mas = tio.ScalarImage(image_paths[0])
    img=flip(mas)
    img.save(output_loc + '/flipped.nii')
    img = rb(mas)
    img.save(output_loc + '/rb.nii')
    img = re(mas)
    img.save(output_loc + '/re.nii')



