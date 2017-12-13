import bug_memory_ComplexEnv as memory
import matplotlib.pyplot as plt
from google_vision_api import requestLabel
import numpy as np
from random import randint

map_size = 100

#Decision Node
def decision(image):
    [label, score] = requestLabel(image) # Output of API converted into a label and a score
    print("Oooooh that's a", label, ". I'm", score, "percent sure!")
    params = memory.getValue(label) # Array of paramaters [alpha, beta,  survival, ex alpha, ex beta]
    # return alpha = coward coefficient; beta = aggressor coefficient
    alpha = params[0]; beta = params[1]; adjustment = params[2]
    return alpha, beta, label, adjustment

def error(totalTime, label):
    # We calculate our error based on the change in Survival Time = Duration 1000 sec
    # if survival time decreases
    #    error = (old time - new time)/total
    #    if error is -ve --> Positive stimulus (Survival Time Has Increased)
    #           increase beta and decrease alpha
    #    if error is +ve --> Negative stimulus (Survival Time Has Decreased)
    #           increase alpha and decrease beta
    params = memory.getValue(label)
    delta = -1*params[2]
    survivalAdjustment = (delta)/(totalTime)
    # if delta = -100  and totalTime = 1000 error should be 0.1
    return survivalAdjustment, delta


def update_error(error,label):
    # Array of paramaters [alpha, beta,  survival, ex alpha, ex beta]
    memory.adjustValue(label, error)

def imgparams(image):
#    ABL=np.zeros((len(image),6))
    ABL=[[],[],[]]
    for i in range(len(image)):
        imagex=0; imagey=0
        while(imagex<15 and imagex>-15) and (imagey<15 and imagey>-15):
            imagex=randint(-map_size + 10, map_size - 10)
            imagey=randint(-map_size + 10, map_size - 10)
        [alpha, beta, label, adjustment] = decision(image[i])
#        ABL[i][0]=alpha
        ABL[i].append(alpha)
#        ABL[i][1]=beta
        ABL[i].append(beta)
#        ABL[i][2]=label
        ABL[i].append(label)
#        ABL[i][3]=adjustment
        ABL[i].append(adjustment)
#        ABL[i][4]=imagex
        ABL[i].append(imagex)
#        ABL[i][5]=imagey
        ABL[i].append(imagey)
        ABL[i].append(0)
    return ABL
