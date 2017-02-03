import music21
import numpy as np

bkey=(1,3,6,8,10)
def is_blackkey(tone):
    ptone=tone%12
    if ptone in bkey:
        return True
    return False

def X2d_to_stream(X,minduration=0.25,base=False):
    strm=music21.stream.Stream()
    bchord=[]
    for x in X:
        s=onehot_to_midin(np.array(x),base=base)
        if s!=bchord:
            strm.append(music21.chord.Chord(s,duration=music21.duration.Duration(minduration)))
        else:
            strm.append(music21.note.Rest(duration=music21.duration.Duration(minduration)))
            s="rest"
        bchord=s
    return strm

def midinum_to_stream(X,minduration=0.25,base=False):
    strm=music21.stream.Stream()
    bchord=[]
    for x in X:
        s=[]
        for xx in x:
            s.append(int(xx))


        if s!=bchord:
            strm.append(music21.chord.Chord(s,duration=music21.duration.Duration(minduration)))
        else:
            strm.append(music21.note.Rest(duration=music21.duration.Duration(minduration)))
            s="rest"
        bchord=s
    return strm

def X2d_to_midinum(X,text=False,base=False):

    res=[]
    for x in X:
        s=onehot_to_midin(np.array(x),base=base)
        hoge=[]
        for ss in s:
            hoge.append(ss)
        if text:
            res.append(midinumber_to_text(hoge))
        else:
            res.append(hoge)
    return res

def getX_midinumber(X):
    for x in X:
        print(x.shape)
        print(x.tolist())
        print(musiclib.onehot_to_midin(x))

def midinumber_to_text(tones_vector):
    res=[]
    for tn in tones_vector:
        nn=music21.note.Note(tn)
        nn=nn.pitch.nameWithOctave
        res.append(nn)
    return res

def basekb_to_num(tones):
    #hoge=[0]*tones[-1]
    #hoge.extend(tones[:(-1)])
    base=tones[-1]
    res=[]
    for k in range(len(tones)):
        if tones[k]==1:
            res.append(int(base+k))
    return res

def get_base(tones):
    for i in range(len(tones)):
        if tones[i]==1:
            return i
    return False

def onehot_to_midin(tones,base=False):
    if base:
        base=int(tones[-1])
    else:
        base=0
    res=[]
    for i in range(len(tones)):
        if tones[i]>0:
            res.append(i+base)
    return res

#def midin_to_onehot(tones):

def min_true_arg(array):
    for k in range(len(array)):
        if array[k]:
            return k
    return False

#bigger than th
def min_arg(array,th):
    for k in range(len(array)):
        if array[k]>=th:
            return k
    return False



def df_to_nparray(df,n_chord):
    X=[]
    y=[]
    for rn,dd in df.iterrows():
        tX=[]
        k=n_chord-1

        if rn<k:
            continue

        if dd[0]!=0:
            tX.append(df.iloc[(rn-k):(rn+1),2:])
        hoge=tX[0]
        X.append(np.array(hoge))
        try:
            y.append(df.iloc[rn+1,3])
        except:
            break

    X=np.array(X)
    return X[:len(y)],np.array(y)
