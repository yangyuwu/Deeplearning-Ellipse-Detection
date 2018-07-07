import urllib.request
import tarfile
import pandas as pd
import numpy as np
from collections import Counter
from matplotlib import pyplot as plt
import random
from PIL import Image
from sklearn.metrics import confusion_matrix
from matplotlib import style
import keras
from keras.layers import Input, Dense
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Dense, Dropout, Concatenate
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras import backend as K
from keras.utils import plot_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import itertools

# plot sampled dataframe rows
def plot_ellipse(df, N_images, test=False):
    select_images = np.random.permutation(len(df))[:N_images]  # permutate images
    random_df = df.loc[select_images]
    if test:
        images = random_df["Testing_images"].values
    else:
        images = random_df["Training_images"].values
    f = plt.figure(figsize=(23,20))
    for i, image in enumerate(images):
        f.add_subplot(1, N_images, i+1)
        x = Image.open(image)
        plt.imshow(x)    
        plt.title("\n".join( ('ellipse:'+random_df.iloc[i]['is_ellipse'] ,'angle:'+str(random_df.iloc[i]['angle']) ,
                    'angle_norm:'+str(random_df.iloc[i]['angle_norm']),
                    'center_x:'+str(random_df.iloc[i]['center_x']) ,'center_y:'+str(random_df.iloc[i]['center_y']),
                    'axis_1:'+str(random_df.iloc[i]['axis_1']) ,'axis_2:'+str(random_df.iloc[i]['axis_2']),
                    'axis_1_new:'+str(random_df.iloc[i]['axis_1_new']) ,'axis_2_new:'+str(random_df.iloc[i]['axis_2_new']),
                    'angle_norm_new:'+str(random_df.iloc[i]['angle_norm_new']))))
        

class ellipse:
    
    def __init__(self, dataset, from_url = True):
        if from_url:
            url = "https://www.dropbox.com/s/3c0y7wpx29l4crp/{}?dl=1".format(dataset) # link for data
            u = urllib.request.urlopen(url)
            data = u.read()
            u.close()
            with open(dataset, "wb") as f:
                f.write(data)
        tar = tarfile.open(dataset)
        tar.extractall()
        tar.close()
    def read_metadata(self):
        # read annotations and coordinates 
        df_train = pd.read_csv('images/train_data.txt', sep=",\s+", engine='python')
        df_test = pd.read_csv('images/test_data.txt', sep=",\s+", engine='python')
        # first 2 columns are merged...
        df_train["Training_images"], df_train["is_ellipse"] = zip(*df_train["Training images: is_ellipse"].str.split().tolist())
        df_train.drop(["Training images: is_ellipse"], 1, inplace = True)
        df_test["Testing_images"], df_test["is_ellipse"] = zip(*df_test["Testing images: is_ellipse"].str.split().tolist())
        df_test.drop(["Testing images: is_ellipse"], 1, inplace = True)
        return df_train, df_test
    
    def add_normalized_angle(self, df):
        idx = np.where(df["axis_1"].values<df["axis_1"].values)
        df["angle_norm"] = df["angle"].values%180
        return df
    
    def set_ax1_bigger_than_ax2(self, df):
        idx = np.where(df["axis_2"].values>df["axis_1"].values)[0]
        df["axis_1_new"] = df["axis_1"].values
        df["axis_2_new"] = df["axis_2"].values
        df["angle_norm_new"] = df["angle_norm"].values
        df.loc[idx,'axis_2_new'] = df.loc[idx,'axis_1'].values
        df.loc[idx,'axis_1_new'] = df.loc[idx,'axis_2'].values
        df.loc[idx,'angle_norm_new'] = (df.loc[idx,'angle_norm'].values+90)%180
        return df

    def get_data(self, df, use_norm_feat = True):
        if use_norm_feat:
            angle = "angle_norm_new"
            axs = "_new"
        else:
            angle = "angle"
            axs = ""
        try:
            images = df["Training_images"].values
        except:
            images = df["Testing_images"].values
        X = np.array([np.array(Image.open(fname)) for fname in images]).astype('float32')
        y = (df["is_ellipse"].values == 'True').astype(float)
        features = df[[angle, "center_x", "center_y", "axis_1{}".format(axs), "axis_2{}".format(axs)]].values.astype(float)
        return X, features, y
    
    def RGB2BW(self, X_train, X_test):
        r, g, b = np.median(X_train[:,:,:,0]), np.median(X_train[:,:,:,1]), np.median(X_train[:,:,:,2])
        std_r, std_g, std_b = X_train[:,:,:,0].std(), X_train[:,:,:,1].std(), X_train[:,:,:,2].std()
        print(r,g,b, '+\-', std_r, std_g, std_b)
        th = np.mean((r,g,b))
        s = np.mean((std_r,std_g,std_b))
        th_high = th+s
        th_low = th-s
        BW_test_tmp = np.logical_or(X_test < th_low, X_test> th_high)
        BW_train_tmp = np.logical_or(X_train < th_low, X_train> th_high)

        BW_test = np.zeros((len(X_test), X_test.shape[1], X_test.shape[2]))
        BW_train = np.zeros((len(X_train), X_train.shape[1], X_train.shape[2]))

        for i in range(len(X_test)):
            BW_test[i,:,:] = np.logical_and(BW_test_tmp[i,:,:,0], BW_test_tmp[i,:,:,1], BW_test_tmp[i,:,:,2])
        for i in range(len(X_train)):
            BW_train[i,:,:] = np.logical_and(BW_train_tmp[i,:,:,0], BW_train_tmp[i,:,:,1], BW_train_tmp[i,:,:,2])
        return BW_train, BW_test

    def build_model(self, rgb_input_shape = (50, 50, 3), bw_input_shape = 2500, feature_shape=5):
        input_bw = Input( shape=(bw_input_shape,) )
        input_rgb = Input( shape = rgb_input_shape )
        # dense model for B&W images
        x_bw = Dense(512, activation='relu')(input_bw)
        x_bw = Dense(64, activation='relu')(x_bw)
        # CNN for RGB images
        x = Conv2D(32, (5,5), name='first_conv', padding='same', activation='relu')(input_rgb)
        x = BatchNormalization(axis=3)(x)
        x = Conv2D(32, (5,5), name='second_conv', activation='relu')(x)
        x = MaxPooling2D()(x)
        x = BatchNormalization(axis=3)(x)
        x = Conv2D(64, (3,3), name='third_conv', activation='relu')(x)
        x = BatchNormalization(axis=3)(x)
        x = Conv2D(64, (3,3), name='forth_conv', activation='relu')(x)
        x = BatchNormalization(axis=3)(x)
        x = Conv2D(128, (2,2), strides=(2,2), name='fifth_conv', activation='relu')(x)
        x = BatchNormalization(axis=3)(x)
        x = Conv2D(256, (2,2), strides=(2,2), name='sixth_conv', activation='relu')(x)
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        # merge bw and rgb activations
        merged_x = Concatenate()([x, x_bw])
        # additional layers for features prediction
        z = Dense(256, activation='relu')(merged_x)
        z = BatchNormalization()(z)
        z = Dense(256, activation='relu')(z)
        z = BatchNormalization()(z)
        z_is_ellipse = Dense(1, activation='sigmoid', name='ellipse')(x)
        z_features = Dense(feature_shape, name='features')(z)

        model = Model([input_bw, input_rgb], [z_features, z_is_ellipse])
        plot_model(model, to_file='Ellipse_Model.png')
        return model