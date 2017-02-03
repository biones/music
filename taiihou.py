'''
TODO
fukyo waon to be junji sinko
'''
import numpy as np
import musiclib

disable_dists=[[1,0],[2,0],[11,9],[10,9],[0,0],[7,7],[6,6],[6,7],
[1,1],[1,2],[2,1],[2,2],[11,11],[10,10],[11,10],[10,11],[11,0],[10,0],[12,12]]
fukyo=[1,2,6,10,11]

#import pdb
def get_dist_two_pich(before_chord,after_chord):
    return [int(np.abs(before_chord[1]-before_chord[0]))%12,int(np.abs(after_chord[0]-after_chord[1]))%12]

def chk_two_voice(before_chord,after_chord,v1,v2,rownum=False,X=False):
    #pdb.set_trace()
    print(before_chord)
    print(after_chord)
    ttones=[before_chord[v1],before_chord[v2]],[after_chord[v1],after_chord[v2]]
    dists=get_dist_two_pich(ttones[0],ttones[1])
    print("dists",dists)
    #input()
    if dists in disable_dists:
        print("disable")
        print(dists)
        #input()
        return False


    if dists[0]==dists[1]:
        b3=X[rownum-2,:]
        print("b3",b3)
        print(tone_abs(b3))
        if dists[0] ==tone_abs([b3[v1],b3[v2]]):
            return False


    if dists[0] in fukyo:
        if not(tone_abs([ttones[0][0],ttones[1][0]])==1 and tone_abs([ttones[0][1],ttones[1][1]])==1):
            return False

    #if dists[1] in (1,2,10,11):
    #    return False

    return True

def tone_abs(d):
    return np.int(np.abs(d[0]-d[1]))%12

import itertools as it
def chk_virtical(chord):
    return True
    for d in it.combinations(chord,2):
        a=tone_abs(d)%12
        if a in (1,2,10,11):
            return False
    return True


class Music_generate:
    def __init__(self,init_chord=False,maxrow=20,number_of_voice=2,whitekey=True,Low=30,High=80):
        self.filenum=0
        self.Low=Low
        self.High=High
        self.only_whitekey=whitekey
        self.X=np.zeros([maxrow,number_of_voice])
        if not init_chord:
            self.X[0,:]=[36,48,52,55,60,64,67,72,76,79,84][:number_of_voice]
        else:
            self.X[0,:]=init_chord


    def generate_chord(self,generate_voice=0,now_chord=False,before_chord=False,rownum=1):
        X=self.X
        if not now_chord:
            before_chord=X[rownum-1,:]
            now_chord=[0]*len(before_chord)


        #print(X)
        D=5
        v=generate_voice
        C=int(before_chord[generate_voice])
        if generate_voice>=1:
            low=max(C-D,now_chord[generate_voice-1],self.Low)
            high=min(now_chord[generate_voice-1]+12,C+D,self.High)
        else:
            low=max(C-D,self.Low)
            high=min(C+D,80)

        rrng=np.random.permutation(range(low+1,high+1))
        #print("rrng",sorted(rrng))
        for now_tone in rrng:
            if self.only_whitekey:
                if musiclib.is_blackkey(now_tone):
                    continue
            now_chord[generate_voice]=now_tone
            print("nc",now_chord)
            #print(now_tone)

            if generate_voice==0:
                self.generate_chord(generate_voice+1,now_chord,before_chord,rownum=rownum)
                continue

            for j in range(generate_voice):
                print("bchk")
                if generate_voice>=1:
                    if not chk_two_voice(before_chord,now_chord,generate_voice,j,rownum=rownum,X=X):
                        break
                print(before_chord)
                print("ncd",now_chord)
                #nput()
                if generate_voice==len(now_chord)-1:
                    if chk_virtical(now_chord):
                        if rownum==len(X)-1:
                            #print("res",X)
                            strm=musiclib.midinum_to_stream(X)
                            #strm.show("musicxml")
                            strm.write('musicxml',fp="xmloutput/gened"+str(self.filenum)+"."+"xml")
                            self.filenum+=1
                            if self.filenum>50:
                                import sys
                                sys.exit()
                            break



                        X[rownum,:]=now_chord
                        self.generate_chord(generate_voice=0,now_chord=False,before_chord=now_chord,rownum=rownum+1)
                        X[rownum,:]=0
                        break
                    continue

                if j==generate_voice-1:
                    print("nextt")
                    self.generate_chord(generate_voice+1,now_chord,before_chord,rownum=rownum)

if __name__=="__main__":
    only_whitekey=True
    maxrow=200
    number_of_voice=2


    X=np.zeros([maxrow,number_of_voice])
    X[0,:]=[36,48,52,55,60,64,67,72,76,79,84][:number_of_voice]
    #X[0,:]=[60,72]

    generate_chord(X,before_chord=X[0,:],rownum=1)
