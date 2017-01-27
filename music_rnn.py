'''
todo:
build model
0.25 beat is 1step by ltsm
'''

from musiclib import *
from music21 import *

import keras
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder as le
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM


class Music:

    def __init__(self):
        self.X2d=[]
        self.raw_y=[]

    def read_xml(self,filename,timedelta,base=True):
        imp=converter.parse(filename)
        self.imp=imp
        self.timedelta=timedelta

        df=get_part_fromstrm(imp.flat,timedelta)
        self.df=df
        self.base=base
        mdf=pd.melt(df,id_vars="time",value_vars="midi")
        X1=[]

        time=0

        smdf=mdf.sort_values(by="time")
        self.smdf=smdf
        raw_y=[]
        while True:
            tones=smdf[smdf.time==time].value
            atones=np.array(tones)
            #print(atones)
            #hoge=np.array(self.chordmidi_to_chordarray(atones))
            hoge=np.array(self.chordmidi_to_chordarray(atones,base=base))
            raw_y.append(self.chordmidi_to_chordarray(atones,base=False))
            X1.append(hoge[0].tolist())
            time+=timedelta

            if time>max(smdf.time):
                break


        self.X2d=X1
        self.raw_y=raw_y
        return X1,raw_y

    def aread_xml(self,filename,timedelta,base=True):
        X,y=self.read_xml(filename,timedelta,base=base)
        self.X2d.extend(X)
        self.raw_y.extend(y)

    def mat_to_rnnX(self,kaisu=5):

        X1=self.X2d
        self.kaisu=kaisu
        y=[]
        #X=np.array([]).reshape(0,len(X1[0]))
        C=len(X1[0])
        X=np.zeros((len(X1),kaisu,C))
        y=np.zeros((len(X1),self.featuredim_y))
        r=0
        for i in range(kaisu,len(X1)):
            #X.append(hoge)
            #X=np.r_[X,np.array(X1[(i-n):i]).reshape(1,X1[0].shape[0])]
            X[r,:,:]=X1[(i-kaisu):i]
            y[r,:]=self.raw_y[i]
            r+=1

        self.X=X
        self.y=y


    def saveXy(self,fname):
        self.outfilex=fname+"rnnX"
        self.outfiley=fname+"rnnY"
        np.save(self.outfilex,self.X)
        np.save(self.outfiley,self.y)

    def loadXy(self):
        self.X=np.load(self.outfilex)
        self.y=np.load(self.outfiley)



    def chordmidi_to_chordarray(self,codemidi,base=False):
        M=40
        self.feature_dim=M+1
        self.featuredim_y=75
        idx=list(zip(["chord"]*M,range(M)))
        idx=pd.MultiIndex.from_tuples(idx)

        if not base:
            cd=pd.DataFrame(0,index=range(1),columns=range(self.featuredim_y))
            for cm in codemidi:
                cd.iloc[0][cm]=1
            return cd

        cd=pd.DataFrame(0,index=range(1),columns=idx)
        try:
            base=min(codemidi)
        except:
            cd.ix[0,"base"]=0
            return cd

        for cm in codemidi:
            cd.iloc[0][("chord",cm-base)]=1

        cd.ix[0,"base"]=base
        return cd


    def fit(self):
        from keras.optimizers import RMSprop
        X,y=self.X,self.y
        #y=np.expand_dims(y, -2)
        out_neurons = y.shape[1]
        model = Sequential()
        model.add(LSTM(10,input_length=X.shape[1],input_dim=X.shape[2],init="zero",
                      activation="relu",stateful=False,go_backwards=False))
        #model.add(LSTM(hidden_neurons,input_shape=(None,None)))

        #model.add(Dense(10,init="zero"))
        #model.add(Activation("relu"))
        #model.add(Activation('sigmoid'))
        #model.add(Dense(10))
        #model.add(Activation("relu"))
        model.add(Dense(out_neurons,init="zero"))
        model.add(Activation('sigmoid'))
        #model.add(Activation('softmax'))

        optimizer = RMSprop(lr=0.3)
        model.compile(loss='binary_crossentropy', optimizer="rmsprop")
        #model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        #model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer)
        #model.compile(loss='mean_squared_error',optimizer="adam")
        model.fit(X,y, nb_epoch=100, validation_split=0.2)
        #model.fit(X,y, nb_epoch=50)

        json_string = model.to_json()
        #open('addrib.json', 'w').write(json_string)
        #model.save_weights("addrib_weight.h5")
        self.model=model


    def read_model(self):
        self.model = model_from_json(open('addrib.json.json').read())
        self.model.load_weights('addrib_weight.h5')

    def print_X2d(self):
        for d in self.X2d:
            print(d)


def read_learn():

    msc=Music()
    fnames=["/Users/iijimasatoshi/Downloads/1079-03.mid","/Users/iijimasatoshi/Downloads/1079-02.mid","/Users/iijimasatoshi/Downloads/bwv773.mid","/Users/iijimasatoshi/Downloads/Prelude2.mid"]

    td=0.25
    kaisu=3
    base=True
    for f in fnames:
        print(len(msc.X2d))
        msc.aread_xml(f,timedelta=td,base=base)

    #msc.aread_xml(fnames[2],timedelta=td,base=base)
    msc.mat_to_rnnX(kaisu=kaisu)
    #msc.saveXy(fnames[0])

    print("made mat")
    #msc.loadXy()
msc.fit()

def generate():
    X=msc.X

    k=100
    #th=0.17
    generate_tnum=2

    Xgen=[]
    for xx in X[k]:
        Xgen.append(onehot_to_midin(xx,base=msc.base))

    #loop for generate
    for i in range(15):
        X_for_pred=[]
        for b in range(1,msc.kaisu+1):
            hoge=msc.chordmidi_to_chordarray(Xgen[-b],base=msc.base)
            X_for_pred.append(hoge.iloc[0].tolist())
        X_for_pred.reverse()
        X_for_pred=np.array([X_for_pred])
        #hoge=[midinumber_to_text(onehot_to_midin(list(X_for_pred[0,i,:]))) for i in range(4)]
        #print(hoge)
        prd=msc.model.predict(X_for_pred)
        th=sorted(prd[0],reverse=True)[generate_tnum]
        base=min_arg(prd[0],th)
        #print(base)
        tn=prd[0]>th
        #tn=tn[base:]+0
        chord_kb=(prd[0]>th)+0
        chord_num=onehot_to_midin(chord_kb,base=msc.base)
        #append
        #chord_num=basekb_to_num(np.r_[tn,base])
        #chord_num=basekb_to_num(np.r_[tn,base])
        #append to df
        chord_num=sorted(list(set(chord_num)))
        Xgen.append(chord_num)
    return Xgen

msc=read_learn()
Xgen=generate()
print(Xgen)
#np.sum(prd)
