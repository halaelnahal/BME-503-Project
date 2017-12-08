from brian2 import *
import matplotlib.pyplot as plt
from random import randint
from main import decision, error
import bug_memory as memory
from google_vision_api import requestLabel


map_size = 100
global imagex, imagey, image_count, bug_plot, image_plot, sr_plot, sl_plot
duration = 5000 * ms
imagex = 50
imagey = 50
image_count = 0

# Sensor neurons
a = 0.02
b = 0.2
c = -65
d = 8

I0 = 2000
tau_ampa = 1.0 * ms
g_synpk = 0.4
g_synmaxval = (g_synpk / (tau_ampa / ms * exp(-1)))

E = -80
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

# Right Motor Neuron
sbr = NeuronGroup(1, sensor_eqs, clock=Clock(0.2 * ms), threshold="v>=30", reset=sensor_reset)  # motor neuron
sbr.v = c
sbr.u = c * b
sbr.imagex = imagex
sbr.imagey = imagey
sbr.mag = 0


# Left Motor Neuron
sbl = NeuronGroup(1, sensor_eqs, clock=Clock(0.2 * ms), threshold="v>=30", reset=sensor_reset)  # motor neuron
sbl.v = c
sbl.u = c * b
sbl.imagex = imagex
sbl.imagey = imagey
sbl.mag = 0

# The virtual bug - may need to adjust these
taum = 4 * ms
base_speed = 75
turn_rate = 50 * Hz

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

# Synapses (sensors communicate with bug motor)



# agressor synapses
w = 2
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
w = 2
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
image_plot = plot(imagex, imagey, 'b*')
# Include Image plot somewhere
sr_plot = plot([0], [0], 'w')  # Just leaving it blank for now
sl_plot = plot([0], [0], 'w')


# Additional update rules (not covered/possible in above eqns)
survival_time = 1000
# Modulate Activity of Coward and Aggressor NN based on Label
syn_ll_c.g_synmax = g_synmaxval*0.15
syn_rr_c.g_synmax = g_synmaxval*0.15
syn_ll_a.g_synmax = g_synmaxval*0.85
syn_rr_a.g_synmax = g_synmaxval*0.85

@network_operation()
def update_positions():
    global imagex, imagey, image_count, survival_time
    sr.x = bug.x + sr.x_disp * sin(bug.angle) + sr.y_disp * cos(bug.angle)
    sr.y = bug.y + - sr.x_disp * cos(bug.angle) + sr.y_disp * sin(bug.angle)

    sl.x = bug.x + sl.x_disp * sin(bug.angle) + sl.y_disp * cos(bug.angle)
    sl.y = bug.y - sl.x_disp * cos(bug.angle) + sl.y_disp * sin(bug.angle)

    #    sr.x = bug.x + sr.x_disp*cos(bug.angle-pi/2) - sr.y_disp*sin(bug.angle-pi/2)
    #    sr.y = bug.y + sr.x_disp*sin(bug.angle-pi/2) + sr.y_disp*cos(bug.angle-pi/2)
    #
    #    sl.x = bug.x + sl.x_disp*cos(bug.angle-pi/2) - sl.y_disp*sin(bug.angle-pi/2)
    #    sl.y = bug.y + sl.x_disp*sin(bug.angle-pi/2) + sl.y_disp*cos(bug.angle-pi/2)

    ##################### Umar Edits ########################
    image = 'Images/7.jpg'
    totalTime = duration/ms

    #####image = 'Images/7.jpg'image = 'Images/7.jpg'######
    if ((bug.x - imagex) ** 2 + (bug.y - imagey) ** 2) < 50:
        [alpha, beta, label] = decision(image)
        syn_ll_c.g_synmax = g_synmaxval*alpha
        syn_rr_c.g_synmax = g_synmaxval*alpha
        syn_ll_a.g_synmax = g_synmaxval*beta
        syn_rr_a.g_synmax = g_synmaxval*beta
        print("The alpha value before backpropagation is", memory.bug_memory[label][0]) #print alpha and beta before adjustment
        print("The beta value before backpropagation is", memory.bug_memory[label][1])
        image_count += 1
        [survivalAdjustment, delta] = error(totalTime, label)
        memory.adjustValue(label,survivalAdjustment)
        print("The alpha value after backpropagation is", memory.bug_memory[label][0]) #print alpha and beta after adjustment
        print("The beta value after backpropagation is", memory.bug_memory[label][1])
        #update_decision(label) # Implement ud()
        survival_time -= delta # Get Current clock time
        print("The survival time is", survival_time, "and has changed by", delta, "during this iteration\n")

        if survival_time <= 0:
            # end Simulation
            print(Clock.t)
            Network.stop()

        imagex = randint(-map_size + 10, map_size - 10)
        imagey = randint(-map_size + 10, map_size - 10)

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


@network_operation(dt=15 * ms)
def update_plot():
    global imagex, imagey, bug_plot, image_plot, sr_plot, sl_plot
    bug_plot[0].remove()
    image_plot[0].remove()
#    sr_plot[0].remove()
#    sl_plot[0].remove()
    bug_x_coords = [bug.x, bug.x - 2 * cos(bug.angle), bug.x - 4 * cos(bug.angle)]  # ant-like body
    bug_y_coords = [bug.y, bug.y - 2 * sin(bug.angle), bug.y - 4 * sin(bug.angle)]
    bug_plot = plot(bug_x_coords, bug_y_coords, 'ko')  # Plot the bug's current position
#    sr_plot = plot([bug.x, sr.x], [bug.y, sr.y], 'k')  # plot the antenna
#    sl_plot = plot([bug.x, sl.x], [bug.y, sl.y], 'k')     #plot the antenna
    image_plot = plot(imagex, imagey,'b*')
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
