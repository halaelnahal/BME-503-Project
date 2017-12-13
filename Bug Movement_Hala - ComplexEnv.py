from brian2 import *
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from random import randint
from main_ComplexEnv import *
import bug_memory_ComplexEnv as memory
from google_vision_api import requestLabel
from PIL import Image
import time

currenttime= time.time()
start_scope()
plt.close('all')

map_size = 100
global imagex, imagey, image_count, bug_plot, image_plot, sr_plot, sl_plot, E
duration = 15000 * ms
image_count = 0

params = {'figure.figsize':(10,5)}
pylab.rcParams.update(params)

# Sensor neurons
a = 0.02
b = 0.2
c = -65
d = 0.5 #8

I0 = 2000 #2000
tau_ampa = 1.0 * ms #1.0
g_synpk = 0.4 #0.4
g_synmaxval = (g_synpk / (tau_ampa / ms * exp(-1)))

image = ['Images/7.jpg', 'Images/1.jpg', 'Images/2.jpg']
ABL=imgparams(image)
    
#if memory.bug_memory[label][0]>0.5:
#    E=0
#elif memory.bug_memory[label][0]<0.5:
#    E=-80
E=-80
    
sensor_eqs = '''
dv/dt=(0.04*v**2+5*v+140-u+I+I_syn)/ms :1
du/dt=(a*(b*v-u))/ms:1
dg_ampa/dt = -g_ampa/tau_ampa+z/ms : 1
dz/dt = (-z/tau_ampa): 1
I_syn=g_ampa*(E-v) : 1
I = I0 / sqrt(((x-imagex)**2+(y-imagey)**2)): 1
y : 1
x : 1
x_disp : 1
y_disp : 1
imagex : 1
imagey : 1
mag :1
E : 1
'''

sensor_reset = '''
v = c
u = u + d
'''

# 2 Right Sensory Neurons ( 1 for coward 1 for aggressor)
sr = NeuronGroup(2, sensor_eqs, clock=Clock(0.2 * ms), threshold="v>=30", reset=sensor_reset)
sr.v = c
sr.u = c * b
sr.x_disp = 5
sr.y_disp = 5
sr.x = sr.x_disp
sr.y = sr.y_disp
sr.imagex = imagex
sr.imagey = imagey
sr.mag = 1
sr.E = E

# 2 Left Sensory Neurons ( 1 for coward 1 for aggressor)
sl = NeuronGroup(2, sensor_eqs, clock=Clock(0.2 * ms), threshold="v>=30", reset=sensor_reset)
sl.v = c
sl.u = c * b
sl.x_disp = -5
sl.y_disp = 5
sl.x = sl.x_disp
sl.y = sl.y_disp
sl.imagex = imagex
sl.imagey = imagey
sl.mag = 1
sl.E = E

# Right Motor Neuron
sbr = NeuronGroup(1, sensor_eqs, clock=Clock(0.2 * ms), threshold="v>=30", reset=sensor_reset)  # motor neuron
sbr.v = c
sbr.u = c * b
sbr.imagex = imagex
sbr.imagey = imagey
sbr.mag = 0
sbr.E = E


# Left Motor Neuron
sbl = NeuronGroup(1, sensor_eqs, clock=Clock(0.2 * ms), threshold="v>=30", reset=sensor_reset)  # motor neuron
sbl.v = c
sbl.u = c * b
sbl.imagex = imagex
sbl.imagey = imagey
sbl.mag = 0
sbl.E = E

# The virtual bug - may need to adjust these
taum = 4 * ms
base_speed = 100
turn_rate = 100 * Hz #originally 50

bug_eqs = '''
#actuation equations go here
dmotorl/dt=  (-motorl/taum)+(I1/ms) :1
dmotorr/dt = (-motorr/taum)+(I2/ms):1
speed = (motorl + motorr)/2 + base_speed : 1
dangle/dt = (motorr - motorl)*turn_rate : 1
dx/dt = speed*cos(angle)*15*Hz : 1
dy/dt = speed*sin(angle)*15*Hz : 1
I1:1
I2:1
'''


bug = NeuronGroup(1, bug_eqs, clock=Clock(0.2 * ms))
bug.motorl = 0
bug.motorr = 0
bug.angle = pi / 2
bug.x = 0
bug.y = 0

bugx=bug.x
bugy=bug.y

def distances(ABL,bugx,bugy):
    distancevec=[]
    xdists=[]
    ydists=[]
    for k in range(len(image)):
        xdists.append(ABL[k][4])
        ydists.append(ABL[k][5])
        distancevec.append(sqrt(((bugx-xdists[k])**2+(bugy-ydists[k])**2)))
    [mindist,index] = min( (distancevec[j],j) for j in range(len(distancevec)) )
    return index

index = distances(ABL,bugx,bugy)
imagex=ABL[index][4]
imagey=ABL[index][5]

# Synapses (sensors communicate with bug motor)
# agressor synapses
w = 2 #originally 2
syn_rr_a = Synapses(sr, sbr, clock=Clock(0.2 * ms), model='''
                g_synmax:1
                ''',
                    on_pre='''
		z+= g_synmax
		''')

syn_rr_a.connect(i=[0], j=[0])
syn_rr_a.g_synmax = g_synmaxval

syn_ll_a = Synapses(sl, sbl, clock=Clock(0.2 * ms), model='''
                g_synmax:1
                ''',
                    on_pre='''
		z+= g_synmax
		''')

syn_ll_a.connect(i=[0], j=[0])
syn_ll_a.g_synmax = g_synmaxval

# coward synapses
w = 2 #originally 2
syn_rr_c = Synapses(sr, sbr, clock=Clock(0.2 * ms), model='''
                g_synmax:1
                ''',
                    on_pre='''
		z+= g_synmax
		''')

syn_rr_c.connect(i=[1], j=[0])
syn_rr_c.g_synmax = 0

syn_ll_c = Synapses(sl, sbl, clock=Clock(0.2 * ms), model='''
                g_synmax:1
                ''',
                    on_pre='''
		z+= g_synmax
		''')

syn_ll_c.connect(i=[1], j=[0])
syn_ll_c.g_synmax = 0

syn_r = Synapses(sbr, bug, clock=Clock(0.2 * ms), on_pre='motorr += w')
syn_r.connect(i=[0], j=[0])
syn_l = Synapses(sbl, bug, clock=Clock(0.2 * ms), on_pre='motorl += w')
syn_l.connect(i=[0], j=[0])

f = figure(1)
bug_plot = plot(bug.x, bug.y, 'ko')
image_plot1=plot(ABL[0][4],ABL[0][5],'b*')
image_plot2=plot(ABL[1][4],ABL[1][5],'k*')
image_plot3=plot(ABL[2][4],ABL[2][5],'r*')
# Include Image plot somewhere
sr_plot = plot([0], [0], 'w')  # Just leaving it blank for now
sl_plot = plot([0], [0], 'w')


# Additional update rules (not covered/possible in above eqns)
survival_time = 1000
# Modulate Activity of Coward and Aggressor NN based on Label
syn_ll_c.g_synmax = g_synmaxval*ABL[index][0]
syn_rr_c.g_synmax = g_synmaxval*ABL[index][0]
syn_ll_a.g_synmax = g_synmaxval*ABL[index][1]
syn_rr_a.g_synmax = g_synmaxval*ABL[index][1]

print('coward', syn_ll_c.g_synmax)
print('coward', syn_rr_c.g_synmax)
print('aggressor', syn_ll_a.g_synmax)
print('aggressor', syn_rr_a.g_synmax)

@network_operation()
def update_positions():
    global imagex, imagey, image_count, survival_time, E, alpha, beta, index, ABL
    sr.x = bug.x + sr.x_disp * sin(bug.angle) + sr.y_disp * cos(bug.angle)
    sr.y = bug.y + - sr.x_disp * cos(bug.angle) + sr.y_disp * sin(bug.angle)

    sl.x = bug.x + sl.x_disp * sin(bug.angle) + sl.y_disp * cos(bug.angle)
    sl.y = bug.y - sl.x_disp * cos(bug.angle) + sl.y_disp * sin(bug.angle)

#    sr.x = bug.x + sr.x_disp*cos(bug.angle-pi/2) - sr.y_disp*sin(bug.angle-pi/2)
#    sr.y = bug.y + sr.x_disp*sin(bug.angle-pi/2) + sr.y_disp*cos(bug.angle-pi/2)
#
#    sl.x = bug.x + sl.x_disp*cos(bug.angle-pi/2) - sl.y_disp*sin(bug.angle-pi/2)
#    sl.y = bug.y + sl.x_disp*sin(bug.angle-pi/2) + sl.y_disp*cos(bug.angle-pi/2)

#    totalTime = duration/ms
    totalTime=1000
    
    index = distances(ABL,bugx,bugy)
    imagex=ABL[index][4]
    imagey=ABL[index][5]
    apiradius=1500
    contactradius=50
    recentindex=3

    if ((bug.x - imagex) ** 2 + (bug.y - imagey) ** 2) > apiradius:
        if ABL[index][6]==1:
            ABL[index][6]=0

    if ((bug.x - imagex) ** 2 + (bug.y - imagey) ** 2) < apiradius and ((bug.x - imagex) ** 2 + (bug.y - imagey) ** 2) > contactradius:
        ABL[index].append(1)
        if ABL[index][6]==0 and recentindex!=index:
            ABL[index][6]=1
            recentindex=index
            [alpha, beta, label, adjustment] = decision(image[index])
            img=Image.open(image[index])
#                if size(imgplot)!=0:
#                    imgplot[0].remove()
            plt.subplot(122)
            imgplot=plt.imshow(img)
            plt.axis('off')
        
            syn_ll_c.g_synmax = g_synmaxval*alpha
            syn_rr_c.g_synmax = g_synmaxval*alpha
            syn_ll_a.g_synmax = g_synmaxval*beta
            syn_rr_a.g_synmax = g_synmaxval*beta
#            print("The alpha value before backpropagation is", memory.bug_memory[label][0]) #print alpha and beta before adjustment
#            print("The beta value before backpropagation is", memory.bug_memory[label][1])
            print("The object is within", apiradius)
                
            if ABL[index][0]>0.5:
                E=0
            elif ABL[index][0]<0.5:
                E=-80
            sr.E = E
            sl.E = E
            sbr.E = E
            sbl.E = E
            print("The E value is", E)
        
    if ((bug.x - imagex) ** 2 + (bug.y - imagey) ** 2) < contactradius:
        ABL[index][6]=0
        [alpha, beta, label, adjustment] = decision(image[index])
        img=Image.open(image[index])
#        if size(imgplot)!=0:
#            imgplot[0].remove()
        plt.subplot(122)
        imgplot=plt.imshow(img)
        plt.axis('off')
        
        syn_ll_c.g_synmax = g_synmaxval*alpha
        syn_rr_c.g_synmax = g_synmaxval*alpha
        syn_ll_a.g_synmax = g_synmaxval*beta
        syn_rr_a.g_synmax = g_synmaxval*beta
        print("The alpha value before backpropagation is", memory.bug_memory[label][0]) #print alpha and beta before adjustment
        print("The beta value before backpropagation is", memory.bug_memory[label][1])

        [survivalAdjustment, delta] = error(totalTime, label)
        memory.adjustValue(label,survivalAdjustment)
        ABL[index][0]=memory.bug_memory[label][0]
        ABL[index][1]=memory.bug_memory[label][1]
        print("The alpha value after backpropagation is", memory.bug_memory[label][0]) #print alpha and beta after adjustment
        print("The beta value after backpropagation is", memory.bug_memory[label][1])
        print(memory.getValue(label))
        
        syn_ll_c.g_synmax = g_synmaxval*memory.bug_memory[label][0]
        syn_rr_c.g_synmax = g_synmaxval*memory.bug_memory[label][0]
        syn_ll_a.g_synmax = g_synmaxval*memory.bug_memory[label][1]
        syn_rr_a.g_synmax = g_synmaxval*memory.bug_memory[label][1]
        
        if ABL[index][0]>0.5:
            E=0
        elif ABL[index][0]<0.5:
            E=-80
        sr.E = E
        sl.E = E
        sbr.E = E
        sbl.E = E
        print("The E value is", E)
        #update_decision(label) # Implement ud()
        survival_time -= delta # Get Current clock time
        print("The survival time is", survival_time, "and has changed by", -1*delta, "during this iteration\n")

        if survival_time <= 0:
            print('Elapsed Time: ' , time.time() - currenttime, 'seconds')

        imagex = randint(-map_size + 10, map_size - 10)
        imagey = randint(-map_size + 10, map_size - 10)
        
        ABL[index][4] = imagex
        ABL[index][5] = imagey
    else:
        E=-80
        sr.E = E
        sl.E = E
        sbr.E = E
        sbl.E = E
        
    if (bug.x < -map_size):
        bug.x = -map_size
        bug.angle = pi - bug.angle
    if (bug.x > map_size):
        bug.x = map_size
        bug.angle = pi - bug.angle
    if (bug.y < -map_size):
        bug.y = -map_size
        bug.angle = -bug.angle
    if (bug.y > map_size):
        bug.y = map_size
        bug.angle = -bug.angle

    sr.imagex = imagex
    sr.imagey = imagey
    sl.imagex = imagex
    sl.imagey = imagey


@network_operation(dt=5 * ms) #originally 15
def update_plot():
    global imagex, imagey, bug_plot, image_plot1, image_plot2, image_plot3, sr_plot, sl_plot, index, ABL
    bug_plot[0].remove()
#    image_plot[0].remove()
    image_plot1[0].remove()
    image_plot2[0].remove()
    image_plot3[0].remove()
#    sr_plot[0].remove()
#    sl_plot[0].remove()
    bug_x_coords = [bug.x, bug.x - 2 * cos(bug.angle), bug.x - 4 * cos(bug.angle)]  # ant-like body
    bug_y_coords = [bug.y, bug.y - 2 * sin(bug.angle), bug.y - 4 * sin(bug.angle)]
    bug_plot = plot(bug_x_coords, bug_y_coords, 'ko')  # Plot the bug's current position
#    sr_plot = plot([bug.x, sr.x], [bug.y, sr.y], 'k')  # plot the antenna
#    sl_plot = plot([bug.x, sl.x], [bug.y, sl.y], 'k')     #plot the antenna

    fig=plt.subplot(121)
#    image_plot = plot(imagex, imagey,'b*')
#    for i in range(len(image)):
#        image_plot=plot(ABL[i][4],ABL[i][5],'b*')
    image_plot1=plot(ABL[0][4],ABL[0][5],'b*')
    image_plot2=plot(ABL[1][4],ABL[1][5],'k*')
    image_plot3=plot(ABL[2][4],ABL[2][5],'r*')
    axis([-100, 100, -100, 100])
    draw()
    # print "."
    pause(0.01)


ML = StateMonitor(sl, ('v', 'I'), record=True)
MR = StateMonitor(sr, ('v', 'I'), record=True)
MB = StateMonitor(bug, ('motorl', 'motorr', 'speed', 'angle', 'x', 'y'), record=True)

run(duration) 

# figure(2)
# plot(ML.t/ms, ML.v[0],'r')
# plot(MR.t/ms, MR.v[0],'b')
# title('v')
# figure(3)
# plot(ML.t/ms, ML.I[0],'r')
# plot(MR.t/ms, MR.I[0],'b')
# title('I')
# figure(4)
# plot(MB.t/ms, MB.motorl[0],'r')
# plot(MB.t/ms, MB.motorr[0],'b')
# title('Motor')


plt.clf()
plt.plot(MB.x[0], MB.y[0])
plt.plot(imagex, imagey)
axis([-100, 100, -100, 100])
title('Path')
show()
plt.plot(MB.x[0], MB.y[0])
plt.plot(imagex, imagey)
axis([-100, 100, -100, 100])
title('Path')
show()
