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


'''
todo:
make class for to deal tone,and model

'''

from musiclib import *
import pandas as pd
import numpy as np
import music21

import keras
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder as le
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM


def sround(v,s):
    return v-v%s


def get_part_fromstrm(strm,timedelta):

    clname=["time","midi"]
    df=[]
    for tn in strm:

        try:
            if type(tn)==music21.chord.Chord:
                midi=[tt.midi for tt in tn.pitches]
            else:
                midi=[tn.midi]
            ofset=sround(tn.offset,timedelta)
            for m in midi:
                d=[ofset,m]
                df.append(d)
        except:
            continue
    df=pd.DataFrame(df,columns=clname)
    return df

class Music:

    def __init__(self):
        self.X2d=[]
        self.raw_y=[]

    def read_xml(self,filename,resolution=2):
        import pretty_midi as pm

        #pm = pretty_midi.PrettyMIDI(resolution=960, initial_tempo=120) #pretty_midiオブジェクトを作ります
        #instrument = pretty_midi.Instrument(0) #instrumentはトラックみたいなものです。
        data=pm.PrettyMIDI(filename, resolution=2)
        X2d=data.get_piano_roll(fs=8)
        X2d=X2d.transpose()
        for i in range(X2d.shape[0]):
            d=X2d[i,:]
            X2d[i,:]=d>0+1-1

        return X2d


    def aread_xml(self,filename,timedelta=False,base=True):
        X2d=self.read_xml(filename)
        self.X2d.extend(X2d)

    def mat_to_rnnX(self,kaisu=5):

        X2d=self.X2d
        self.kaisu=kaisu
        y=[]
        #X=np.array([]).reshape(0,len(X1[0]))
        C=len(X2d[0])
        X=np.zeros((len(X2d),kaisu,C))
        y=np.zeros((len(X2d),len(X2d[0])))
        r=0
        for i in range(kaisu,len(X2d)):
            X[r,:,:]=X2d[(i-kaisu):i]
            y[r,:]=X2d[i]
            r+=1

        self.X=X
        self.y=y

    def saveXy(self,fname):
        self.outfilex=fname+"X"
        self.outfiley=fname+"Y"
        np.save(self.outfilex,self.X)
        np.save(self.outfiley,self.y)

    def loadXy(self,fname):
        self.outfilex=fname+"X"
        self.outfiley=fname+"Y"
        self.X=np.load(self.outfilex+".npy")
        self.y=np.load(self.outfiley+".npy")


    def chordmidi_to_chordarray(self,codemidi,base=False):
        M=40
        self.feature_dim=M+1
        self.featuredim_y=128
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

    #def chordmidi_to_chordarray(codemidi,base=False):


    def fit(self,epoch):
        from keras.optimizers import RMSprop
        from keras.callbacks import EarlyStopping
        X,y=self.X,self.y
        #y=np.expand_dims(y, -2)
        out_neurons = y.shape[1]
        print("shape of X",X.shape)
        model = Sequential()
        model.add(LSTM(20,input_shape=(X.shape[1],X.shape[2])))
                        #batch_input_shape=(1,X.shape[1],X.shape[2])))
        #model.add(LSTM(hidden_neurons,input_shape=(None,None)))
        #model.add(LSTM(10,input_length=X.shape[1],init="zero",
        #            activation="relu",stateful=False,go_backwards=False))
        #model.add(Dense(20))
        model.add(Activation("relu"))
        model.add(Dense(10))
        model.add(Activation("relu"))
        #model.add(Activation('sigmoid'))
        #model.add(Dense(10))
        #model.add(Activation("relu"))
        #model.add(Dense(out_neurons,init="zero"))
        model.add(Dense(out_neurons))
        model.add(Activation('sigmoid'))
        #model.add(Activation('softmax'))
        #optimizer = RMSprop(lr=0.1)
        model.compile(loss='binary_crossentropy', optimizer="adam",metrics=['accuracy'],shuffle=True)
        #model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=['accuracy'])
        #model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer)
        #model.compile(loss='mean_squared_error',optimizer="adam")
        #early_stopping = EarlyStopping(monitor='accuracy', patience=2)
        self.history=model.fit(X,y, nb_epoch=epoch, validation_split=0.2,batch_size=30)#,callbacks=[early_stopping])
        #json_string = model.to_json()
        #open('addrib.json', 'w').write(json_string)
        #model.save_weights("addrib_weight.h5")
        self.model=model

    def fit_auto_encorder(self,epoch):
        from keras.optimizers import RMSprop
        from keras.callbacks import EarlyStopping
        from keras.models import Model
        from keras.layers import Input, Convolution2D

        X,y=self.X,self.y

        a = Input(shape=(1,X.shape[1],X.shape[2]))
        x=LSTM(20,input_length=X.shape[1],input_dim=X.shape[2])
                        #batch_input_shape=(1,X.shape[1],X.shape[2])))
        #model.add(LSTM(hidden_neurons,input_shape=(None,None)))
        #model.add(LSTM(10,input_length=X.shape[1],init="zero",
        #            activation="relu",stateful=False,go_backwards=False))
        #model.add(Dense(20))
        x=Dense(20,activation="relu")(x)
        decoded=Dense(X.shape[2],activation="sigmoid")(x)
        ae = Model(a,decoded)
        ae.compile(loss='binary_crossentropy', optimizer="adam",metrics=['accuracy'],shuffle=True)
        #model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=['accuracy'])
        #model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer)
        #model.compile(loss='mean_squared_error',optimizer="adam")
        #early_stopping = EarlyStopping(monitor='accuracy', patience=2)
        self.history=ae.fit(X,X, nb_epoch=epoch, validation_split=0.2,batch_size=20)#,callbacks=[early_stopping])
        #json_string = model.to_json()
        #open('addrib.json', 'w').write(json_string)
        #model.save_weights("addrib_weight.h5")
        self.hidden=x
        self.model=model



    def read_model(self):
        self.model = model_from_json(open('addrib.json.json').read())
        self.model.load_weights('addrib_weight.h5')

    def print_X2d(self):
        for d in self.X2d:
            print(d)





def read_learn():

    msc=Music()
    fnames=["/Users/iijimasatoshi/Downloads/Prelude2.mid"]

    kaisu=5
    base=True
    for f in fnames:
        print(len(msc.X2d))
        msc.aread_xml(f,base=base)

    msc.mat_to_rnnX(kaisu=kaisu)
    #msc.saveXy(fnames[0])

    print("made mat")

    return msc



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

if __name__=="main":
    msc=read_learn()
    msc.fit(10)
    #Xgen=generate()
    #print(Xgen)
    #np.sum(prd)
