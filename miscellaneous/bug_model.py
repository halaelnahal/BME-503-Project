from brian2 import *
import matplotlib.pyplot as plt

# Map Size
map_size = 100
defaultclock.dt = 0.1*ms

# Global Variables
global foodx, foody, food_count, bug_plot, food_plot, sr_plot, sl_plot

# Initialize Additional global variables to keep track of Toxin
global toxinx, toxiny, toxin_count, toxin_plot, tr_plot, tl_plot

# Food Variables
foodx = 50; foody = 50; food_count = 0

# Toxin Variables
toxinx = -25; toxiny = -50; toxin_count = 0

# Sensor neurons (Izhekevich Model)
a = 0.02; b = 0.2; c = -65; d = 0.5

# The virtual bug - may need to adjust these
taum = 4*ms
base_speed = 9.5
turn_rate = 5*Hz

# Synapse Parameters
Cm=1.0*ufarad/cm**2
area=20000*umetre**2
I0 = 750
g_synpk=0.12*nsiemens
g_synmax=(g_synpk/(taum/ms*exp(-1)))

# Differential Equations for food sensor, toxin sensor, and bug
sensor_eqs = '''
dv/dt = (0.04*(v)*(v) + 5*(v) + 140- u)/ms
        + (I/mV)/(Cm*area):  1
du/dt = a*(b*v - u)/ms : 1

y : 1
x : 1
x_disp : 1
y_disp : 1

foodx : 1
foody : 1


I = I0*nA /sqrt((x-foodx)**2+(y-foody)**2): amp

'''

sensor_eqs_toxin= '''
dv/dt = (0.04*(v)*(v) + 5*(v) + 140- u)/ms
        + (I/mV)/(Cm*area):  1
du/dt = a*(b*v - u)/ms : 1

y : 1
x : 1
x_disp : 1
y_disp : 1

toxinx : 1
toxiny : 1

I = I0*nA /sqrt((x-toxinx)**2+(y-toxiny)**2): amp
'''

sensor_reset = '''
v = c
u = u + d
'''


bug_eqs = '''
dv/dt = (0.04*(v)*(v) + 5*(v) + 140- u)/ms :  1
du/dt = a*(b*v - u)/ms : 1

motorl= g_food_r/nS + g_toxin/nS + 1/2*(1/v): 1
motorr= g_food/nS + g_toxin_r/nS + 1/2*(1/v): 1
speed = (motorl + motorr)/2 + base_speed : 1
dangle/dt = (motorr - motorl)*turn_rate : 1
dx/dt = speed*cos(angle)*15*Hz : 1
dy/dt = speed*sin(angle)*15*Hz : 1

dsa/dt = -sa/taum: siemens
dsb/dt = -sb/taum: siemens
dsc/dt = -sc/taum: siemens
dsd/dt = -sd/taum: siemens

dg_toxin/dt=-g_toxin/taum + sa/ms : siemens
dg_toxin_r/dt=-g_toxin_r/taum + sb/ms : siemens

dg_food/dt=-g_food/taum + sc/ms : siemens
dg_food_r/dt=-g_food_r/taum + sd/ms : siemens
'''



# Initiate all Neuron Groups: total 5 neurons
sr = NeuronGroup(1, sensor_eqs, clock=Clock(0.2*ms),
                    threshold = "v>=30", reset = sensor_reset, method = "euler")
sr.v = c
sr.u = c*b
sr.x_disp = 5
sr.y_disp = 5
sr.x = sr.x_disp
sr.y = sr.y_disp
sr.foodx = foodx
sr.foody = foody


sl = NeuronGroup(1, sensor_eqs, clock=Clock(0.2*ms),
                threshold = "v>=30", reset = sensor_reset, method = "euler")
sl.v = c
sl.u = c*b
sl.x_disp = -5
sl.y_disp = 5
sl.x = sl.x_disp
sl.y = sl.y_disp
sl.foodx = foodx
sl.foody = foody


tr = NeuronGroup(1, sensor_eqs_toxin, clock=Clock(0.2*ms),
                    threshold = "v>=30", reset = sensor_reset, method = "euler") #toxin neuron
tr.v = c
tr.u = c*b
tr.x_disp = 5
tr.y_disp = 5
tr.x = tr.x_disp
tr.y = tr.y_disp
tr.toxinx = toxinx
tr.toxiny = toxiny


tl = NeuronGroup(1, sensor_eqs_toxin, clock=Clock(0.2*ms),
                    threshold = "v>=30", reset = sensor_reset, method = "euler") #toxin neuron
tl.v = c
tl.u = c*b
tl.x_disp = -5
tl.y_disp = 5
tl.x = tl.x_disp
tl.y = tl.y_disp
tl.toxinx = toxinx
tl.toxiny = toxiny



bug = NeuronGroup(1, bug_eqs, clock=Clock(0.2*ms), method = "euler")
# bug.motorl = 0
# bug.motorr = 0
bug.angle = pi/2
bug.x = 0
bug.y = 0
bug.v = c
bug.u = c*b

################################################################################

# Synapses (sensors communicate with bug motor)

# Food Left and Right Sensory to Motor Synapses
syn_r=Synapses(sr, bug, clock=sr.clock, model='''
                w: siemens
                ''',
		on_pre='''
		g_food_r += w
		''')
syn_r.connect(i=[0],j=[0])
syn_r.w=g_synmax/(defaultclock.dt/ms)*10

syn_l=Synapses(sl, bug, clock=sl.clock, model='''
                w: siemens
                ''',
		on_pre='''
		g_food += w
		''')
syn_l.connect(i=[0],j=[0])
syn_l.w=g_synmax/(defaultclock.dt/ms)*10



# Toxin Left and Right Sensory to Motor Synapses

syn_tr = Synapses(tr, bug, clock=tr.clock, model='''
                w: siemens
                ''',
		on_pre='''
		g_toxin_r += w
		''')
syn_tr.connect(i=[0],j=[0])
syn_tr.w=g_synmax/(defaultclock.dt/ms)*10

syn_tl = Synapses(tl, bug, clock=tl.clock, model='''
                w: siemens
                ''',
		on_pre='''
		g_toxin += w
		''')
syn_tl.connect(i=[0],j=[0])
syn_tl.w=g_synmax/(defaultclock.dt/ms)*10



f = figure(1)
bug_plot = plot(bug.x, bug.y, 'bo')
food_plot = plot(foodx, foody, 'g*')
toxin_plot = plot(toxinx, toxiny, 'r*')

# Just leaving it blank for now
sr_plot = plot([0], [0], 'w')
sl_plot = plot([0], [0], 'w')
tr_plot = plot([0], [0], 'w')
tl_plot = plot([0], [0], 'w')
# Additional update rules (not covered/possible in above eqns)

# Array to keep track of all coordinates of the bug
x_plot = []
y_plot = []

# This block of code updates the position of the bug and
# food and makes the appropriate rotations
@network_operation()
def update_positions():
    global foodx, foody, food_count
    global toxinx, toxiny, toxin_count

    sr.x = bug.x + sr.x_disp*cos(bug.angle-pi/2) - sr.y_disp*sin(bug.angle-pi/2)
    sr.y = bug.y + sr.x_disp*sin(bug.angle-pi/2) + sr.y_disp*cos(bug.angle-pi/2)

    sl.x = bug.x + sl.x_disp*cos(bug.angle-pi/2) - sl.y_disp*sin(bug.angle-pi/2)
    sl.y = bug.y + sl.x_disp*sin(bug.angle-pi/2) + sl.y_disp*cos(bug.angle-pi/2)

    tr.x = bug.x + tr.x_disp*cos(bug.angle-pi/2) - tr.y_disp*sin(bug.angle-pi/2)
    tr.y = bug.y + tr.x_disp*sin(bug.angle-pi/2) + tr.y_disp*cos(bug.angle-pi/2)

    tl.x = bug.x + tl.x_disp*cos(bug.angle-pi/2) - tl.y_disp*sin(bug.angle-pi/2)
    tl.y = bug.y + tl.x_disp*sin(bug.angle-pi/2) + tl.y_disp*cos(bug.angle-pi/2)

    if ((bug.x-toxinx)**2+(bug.y-toxiny)**2) < 10:
    	toxin_count += 1
    	toxinx = randint(-map_size+10, map_size-10)
    	toxiny = randint(-map_size+10, map_size-10)

    if ((bug.x-foodx)**2+(bug.y-foody)**2) < 10:
    	food_count += 1
    	foodx = randint(-map_size+10, map_size-10)
    	foody = randint(-map_size+10, map_size-10)

    # Rebound upon encountering the edge
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

    sr.foodx = foodx
    sr.foody = foody
    sl.foodx = foodx
    sl.foody = foody

    tr.toxinx = toxinx
    tr.toxiny = toxiny
    tl.toxinx = toxinx
    tl.toxiny = toxiny

    # Array to keep track of all coordinates of the bug
    x_plot.append(float(bug.x/1.0))
    y_plot.append(float(bug.y/1.0))

# this block of code updates the plots so
# you can see the bug and food move
@network_operation(dt=2*ms)
def update_plot():
    global foodx, foody, bug_plot, food_plot, sr_plot, sl_plot
    global toxinx, toxiny, toxin_count, toxin_plot, tr_plot, tl_plot
    bug_plot[0].remove()
    food_plot[0].remove()
    toxin_plot[0].remove()
    sr_plot[0].remove()
    sl_plot[0].remove()

    bug_x_coords = [bug.x, bug.x-2*cos(bug.angle), bug.x-4*cos(bug.angle)]
                    # ant-like body
    bug_y_coords = [bug.y, bug.y-2*sin(bug.angle), bug.y-4*sin(bug.angle)]
    # Plot the bug's current position
    bug_plot = plot(bug_x_coords, bug_y_coords, 'yo')
    sr_plot = plot([bug.x, sr.x], [bug.y, sr.y], 'b')
    sl_plot = plot([bug.x, sl.x], [bug.y, sl.y], 'r')

    food_plot = plot(foodx, foody, 'b*')
    toxin_plot = plot(toxinx, toxiny, 'r*')
    axis([-100,100,-100,100])
    draw()
    #print "."
    pause(0.001)

# ML = StateMonitor(sl, ('v', 'I'), record=True)
# MR = StateMonitor(sr, ('v', 'I'), record=True)
# MB = StateMonitor(bug, ('motorl', 'motorr', 'speed', 'angle', 'x', 'y'), record = True)

run(2000*ms,report='text')
show()
