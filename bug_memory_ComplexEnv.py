# Bug training data process

# Hash table for bug memory, with each food/predator has its own alpha and beta values
bug_memory = {}
# predator [alpha, beta, survival, expected alpha, expected beta]
bug_memory['spider'] = [0, 1, -100, 1, 0]
# bug_memory['spider'] = [0.49, 0.51, -100, 1, 0]
bug_memory['bird'] = [0.49, 0.51, -50, 1, 0]
bug_memory['bat'] = [0.49, 0.51, -100, 1, 0]
bug_memory['snake'] = [0.49, 0.51, -20, 1, 0]
bug_memory['chicken'] = [0.49, 0.51, -50, 1, 0]
bug_memory['duck'] = [0.49, 0.51, -50, 1, 0]

# food
bug_memory['sunflower'] = [1, 0, 100, 0, 1]
bug_memory['strawberry'] = [0.49, 0.51, 100, 0, 1]
bug_memory['rose'] = [0.49, 0.51, 20, 0, 1]
bug_memory['leaf'] = [1, 0, 20, 0, 1]
bug_memory['orange'] = [0.49, 0.51, 10, 0, 1]

def getValue(label):
    return bug_memory[label]

def adjustValue(label, survivalAdjustment): #pseudo backpropagation
    params = bug_memory.get(label)
    if survivalAdjustment < 0:
        if (params[0]-abs(survivalAdjustment)<0 or params[1]+abs(survivalAdjustment)>1):
            params[0] = params[3]# expected alpha
            params[1] = params[4] # expected beta
        else:
            params[0] -= abs(survivalAdjustment) # decrease alpha
            params[1] += abs(survivalAdjustment) # increase beta
    elif survivalAdjustment > 0:
        if (params[0]+abs(survivalAdjustment)>1 or params[1]-abs(survivalAdjustment)<0):
            params[0] = params[3] # expected alpha
            params[1] = params[4] # expected beta
        else:
            params[0] += abs(survivalAdjustment) # increase alpha
            params[1] -= abs(survivalAdjustment) # decrease beta
    bug_memory[label][0] = params[0]
    bug_memory[label][1] = params[1]
