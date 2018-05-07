# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 10:07:45 2017

@author: Common
"""
import csv
from PIL import Image, ImageOps
import numpy as np
import os
import pdb

qwe=pdb.set_trace    


def createSetFromCSV(csvfilename):  
    x_set,y_set=[],[]
    with open(csvfilename) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:        
            x,y=[],[]
            col=list(row.keys())
            for i in range(len(col)):   
                if(col[i].count('X')>0):
                    x.append(float(row[col[i]]))
                elif(col[i].count('Y')>0):
                    y.append(int(row[col[i]]))
            x_set.append(x)
            y_set.append(y)  
    return x_set,y_set

def csvOutput(listX,name):        
    outputstr='' 
    for i in listX:
        try:
            for j in range(len(i)):
                outputstr+=str(i[j])
                if j<len(i)-1:outputstr+=','
        except:
            outputstr+=str(i)
        outputstr+='\n'        
    
    with open(name, "w") as text_file:
        text_file.write(outputstr)


def standardizeImg(im,size):  
    im=ImageOps.expand(im,int(np.absolute(im.size[1]-im.size[0])/2),(0,0,0))    
    im=ImageOps.fit(im,size)
    return im

def createDatasets(path,size=[100,100]):
    x,y=[],[]
    classCount=len(os.listdir(path))
    for classNo in os.listdir(path):                    
        pathClass=path+'//'+classNo
        for item in os.listdir(pathClass):            
            pathClassItem=pathClass+'//'+item
            im=np.array(standardizeImg(Image.open(pathClassItem),size),np.float32)           
            x.append(im)    
            
            y_=np.zeros(classCount,np.int16)            
            y_[int(classNo)]=1
            y.append(y_)
    x=np.array(x)   
    y=np.array(y)
    return x,y

#-to convert array to img, use Image.fromarray(np.uint8((obj)*255)).show()