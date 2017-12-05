# Bug training data process

# Hash table for bug memory, with each food/predator has its own alpha and beta values
bug_memory = {}
# predator [alpha, beta, survival, expected alpha, expected beta]
bug_memory['spider'] = [0.49, 0.51, -100, 1, 0]
# bug_memory['spider'] = [0.49, 0.51, -100, 1, 0]
bug_memory['bird'] = [0.49, 0.51, -50, 1, 0]
bug_memory['bat'] = [0.49, 0.51, -100, 1, 0]
bug_memory['snake'] = [0.49, 0.51, -20, 1, 0]
bug_memory['chicken'] = [0.49, 0.51, -50, 1, 0]
bug_memory['duck'] = [0.49, 0.51, -50, 1, 0]

# food
bug_memory['sunflower'] = [0.49, 0.51, 100, 0, 1]
bug_memory['strawberry'] = [0.49, 0.51, 100, 0, 1]
bug_memory['rose'] = [0.49, 0.51, 20, 0, 1]
bug_memory['leaf'] = [0.49, 0.51, 20, 0, 1]
bug_memory['orange'] = [0.49, 0.51, 10, 0, 1]

def getValue(key):
    return bug_memory[key]

def adjustValue(key, error): #pseudo backpropagation
    params = bug_memory.get(key)
    if error < 0:
        if (params[0]-abs(error)<0 or params[1]+abs(error)>1):
            error *= 0.1
        params[0] -= abs(error) # decrease alpha
        params[1] += abs(error) # increase beta
    elif error > 0:
        if (params[0]+abs(error)>1 or params[1]-abs(error)<0):
            error *= 0.1
        params[0] += abs(error) # increase alpha
        params[1] -= abs(error) # decrease beta
    bug_memory[key] = params
    #!!!!!!!!!! Dictionary Values Not Being Stored !!!!!!!!!!!!!!!#
