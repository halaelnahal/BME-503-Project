import bug_memory as memory
import matplotlib.pyplot as plt
from google_vision_api import requestLabel

#Decision Node
def decision(image):
    [label, score] = requestLabel(image) # Output of API converted into a label and a score
    print("Oooooh that's a", label, ". I'm", score, "percent sure!")
    params = memory.getValue(label) # Array of paramaters [alpha, beta,  survival, ex alpha, ex beta]
    # return alpha = coward coefficient; beta = aggressor coefficient
    alpha = params[0]; beta = params[1]
    return alpha, beta, label

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
    print(error)
    # if delta = -100  and totalTime = 1000 error should be 0.1
    return survivalAdjustment, params[2]


def update_error(error,label):
    # Array of paramaters [alpha, beta,  survival, ex alpha, ex beta]
    memory.adjustValue(label, error)



image = 'Images/7.jpg'
alpha, beta, label = decision(image)
print('alpha: ', alpha)
print('beta: ', beta)
print('label: ', label)
print('####################  Starting Simulation ############################')
totalTime = 1000;
[survivalAdjustment, timeAdjustment]= error(totalTime, label)
print('Error: ', survivalAdjustment, timeAdjustment)
update_error(survivalAdjustment,label)
params = memory.getValue(label)
print('alpha: ', params[0])
print('beta: ', params[1])
print('######iteration 2#############')
print(memory.getValue('spider'))
update_error(survivalAdjustment,label)
print(memory.getValue('spider'))
