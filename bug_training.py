# Bug training data process

# Hash table for bug memory, with each food/predator has its own alpha and beta values
bug_memory = {}
# predator [alpha, beta, survival, expected alpha, expected beta]
bug_memory['spider'] = [0.49, 0.51, -100, 1, -1]
bug_memory['bird'] = [0.49, 0.51, -50, 1, -1]
bug_memory['bat'] = [0.49, 0.51, -100, 1, -1]
bug_memory['snake'] = [0.49, 0.51, -20, 1, -1]
bug_memory['chicken'] = [0.49, 0.51, -50, 1, -1]
bug_memory['duck'] = [0.49, 0.51, -50, 1, -1]

# food
bug_memory['sunflower'] = [0.49, 0.51, 100, -1, 1]
bug_memory['strawberry'] = [0.49, 0.51, 100, -1, 1]
bug_memory['rose'] = [0.49, 0.51, 20, -1, 1]
bug_memory['leaf'] = [0.49, 0.51, 20, -1, 1]
bug_memory['orange'] = [0.49, 0.51, 10, -1, 1]

from google_vision_api import requestLabel

fileName = '/Users/shishen/Desktop/spider1.jpeg'
[label, score] = requestLabel(fileName)
print('requested',label)
print(score)

def getValue(key):
    return bug_memory[key]
    
def adjustValue(key,value):
    bug_memory[key] = value