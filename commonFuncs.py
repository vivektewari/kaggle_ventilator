import pandas as pd
import numpy as np
from collections import namedtuple

def objectTodf( objectList):
    lst = []
    for v in objectList:
        lst.append(v.__dict__)
    final=pd.DataFrame(lst)
    return final
def dfToObject(df,class1):
    cols=df.columns
    return [class1(*[row[x] for x in cols ]) for  index,row in df.iterrows()]
class packing():
    codeList={'list':'01','MultiValue':'01','dict':'02','str':'10','int':'11','float':'12','DSfloat':'12','float64':'12','int64':'11','ndarray':'03','na':'99','NoneType':'98'}
    dictFuncs={'11':int,'12':float,'10':str,'99':lambda x:"",'98':lambda x:None}
    #inv_codeList = {v: k for k, v in codeList.items()}
    def __init__(self):
        pass
    def strToList(string1):
        return string1.strip('][\n').split(' ')
    def listToStr(list1):
        str1="|"
        for l in list1:
            str1+=";"+str(l)
        str1+="|"
        str1=str1.replace('|;','|')
        return  str1
    def dictToStr(dict1):
        str1=""
        str1+=packing.listToStr(dict1.keys())
        str1+=packing.listToStr(dict1.values())
        return str1
    def pack(x):
        #print(x)
        type1=type(x).__name__
        type2='na'
        type3='na'
        r=None

        if type1 in  ['list','ndarray']:
            type2=type(x[0]).__name__
            type3=None
            r=packing.codeList[type1]+packing.codeList[type2]+packing.listToStr(x)
        elif type1 in  ['dict']:
            keys=list(x.keys())
            if len(keys)>0:
                type2=type(list(x.keys())[0]).__name__
                type3 = type(list(x.values())[0]).__name__
            r=packing.codeList[type1] + packing.codeList[type2] +packing.codeList[type3]+ packing.dictToStr(x)

        elif type1 in ['MultiValue']:
            x=list(x)
            type2 = type(x[0]).__name__
            type3 = None
            r = packing.codeList[type1] + packing.codeList[type2] + packing.listToStr(x)
        elif type1 in ['str','int','float','DSfloat','float64','int64']:
            r=x

        else:
            raise Warning("type not identofied")
        return r
    def unpack(x):

        try:
            str1=x.split('|')
        except :
            return x
        if len(str1)==1:return x
        code=str1[0]
        types=[]
        for i in range(int(len(code)/2)):
            types.append(code[i*2:(i+1)*2])
        if types[0] in ['01','03']:
            list1=str1[1].split(';')
            func1=packing.dictFuncs[types[1]]
            finalList=[]
            for l in list1:
                try:
                    finalList.append(func1(l))
                except:
                    finalList.append(l)

            if types[0]=='03':finalList=np.array(finalList)
            r=finalList
        elif types[0] =='02':
             list1 = str1[1].split(';')
             list2 = str1[3].split(';')
             func1 = packing.dictFuncs[types[1]]
             func2 = packing.dictFuncs[types[2]]
             finalDict = {}
             for i in range(0,len(list1)):

                finalDict[func1(list1[i])]=func2(list2[i])
             r = finalDict

             if list(r.keys())[0]=="":r={}
        return r



if __name__=='__main__':
    d=packing.pack([1.0,2,3])
    e=packing.pack({1.0:2,2:4})
    j=packing.unpack(e)
    k=packing.unpack(d)
    b=0
