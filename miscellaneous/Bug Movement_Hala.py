from brian2 import *
import matplotlib.pyplot as plt
from random import randint

map_size = 100
global foodx, foody, food_count, bug_plot, food_plot, sr_plot, sl_plot, food_color
duration = 5000 * ms
foodx = 50
foody = 50
food_count = 0
food_color = 'g*'

# Sensor neurons
a = 0.02
b = 0.2
c = -65
d = 8

I0 = 2000
tau_ampa = 1.0 * ms
g_synpk = 0.4
g_synmaxval = (g_synpk / (tau_ampa / ms * exp(-1)))

E = 0
sensor_eqs = '''
dv/dt=(0.04*v**2+5*v+140-u+I+I_syn)/ms :1
du/dt=(a*(b*v-u))/ms:1
dg_ampa/dt = -g_ampa/tau_ampa+z/ms : 1
dz/dt = (-z/tau_ampa): 1
I_syn=g_ampa*(E-v) : 1
I = I0 / sqrt(((x-foodx)**2+(y-foody)**2)): 1
y : 1
x : 1
x_disp : 1
y_disp : 1
foodx : 1
foody : 1
mag :1

'''

sensor_reset = '''
v = c
u = u + d
'''

sr = NeuronGroup(2, sensor_eqs, clock=Clock(0.2 * ms), threshold="v>=30", reset=sensor_reset)
sr.v = c
sr.u = c * b
sr.x_disp = 5
sr.y_disp = 5
sr.x = sr.x_disp
sr.y = sr.y_disp
sr.foodx = foodx
sr.foody = foody
sr.mag = 1

sl = NeuronGroup(2, sensor_eqs, clock=Clock(0.2 * ms), threshold="v>=30", reset=sensor_reset)
sl.v = c
sl.u = c * b
sl.x_disp = -5
sl.y_disp = 5
sl.x = sl.x_disp
sl.y = sl.y_disp
sl.foodx = foodx
sl.foody = foody
sl.mag = 1

sbr = NeuronGroup(1, sensor_eqs, clock=Clock(0.2 * ms), threshold="v>=30", reset=sensor_reset)  # motor neuron
sbr.v = c
sbr.u = c * b
sbr.foodx = foodx
sbr.foody = foody
sbr.mag = 0

sbl = NeuronGroup(1, sensor_eqs, clock=Clock(0.2 * ms), threshold="v>=30", reset=sensor_reset)  # motor neuron
sbl.v = c
sbl.u = c * b
sbl.foodx = foodx
sbl.foody = foody
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
food_plot = plot(foodx, foody, 'b*')
sr_plot = plot([0], [0], 'w')  # Just leaving it blank for now
sl_plot = plot([0], [0], 'w')


# Additional update rules (not covered/possible in above eqns)

@network_operation()
def update_positions():
    global foodx, foody, food_count, FC, food_color, E
    sr.x = bug.x + sr.x_disp * sin(bug.angle) + sr.y_disp * cos(bug.angle)
    sr.y = bug.y + - sr.x_disp * cos(bug.angle) + sr.y_disp * sin(bug.angle)

    sl.x = bug.x + sl.x_disp * sin(bug.angle) + sl.y_disp * cos(bug.angle)
    sl.y = bug.y - sl.x_disp * cos(bug.angle) + sl.y_disp * sin(bug.angle)

    #    sr.x = bug.x + sr.x_disp*cos(bug.angle-pi/2) - sr.y_disp*sin(bug.angle-pi/2)
    #    sr.y = bug.y + sr.x_disp*sin(bug.angle-pi/2) + sr.y_disp*cos(bug.angle-pi/2)
    #
    #    sl.x = bug.x + sl.x_disp*cos(bug.angle-pi/2) - sl.y_disp*sin(bug.angle-pi/2)
    #    sl.y = bug.y + sl.x_disp*sin(bug.angle-pi/2) + sl.y_disp*cos(bug.angle-pi/2)


    FoodColor = randint(1, 2)
    if ((bug.x - foodx) ** 2 + (bug.y - foody) ** 2) < 16:
        food_count += 1
        foodx = randint(-map_size + 10, map_size - 10)
        foody = randint(-map_size + 10, map_size - 10)
        if FoodColor == 1:
            food_color = 'r*'
            syn_ll_c.g_synmax = g_synmaxval
            syn_rr_c.g_synmax = g_synmaxval
            syn_ll_a.g_synmax = 0
            syn_rr_a.g_synmax = 0
        elif FoodColor == 2:
            food_color = 'g*'
            syn_ll_c.g_synmax = 0
            syn_rr_c.g_synmax = 0
            syn_ll_a.g_synmax = g_synmaxval
            syn_rr_a.g_synmax = g_synmaxval

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


@network_operation(dt=15 * ms)
def update_plot():
    global foodx, foody, bug_plot, food_plot, sr_plot, sl_plot, FC, food_color
    bug_plot[0].remove()
    food_plot[0].remove()
    #sr_plot[0].remove()
    # sl_plot[0].remove()
    bug_x_coords = [bug.x, bug.x - 2 * cos(bug.angle), bug.x - 4 * cos(bug.angle)]  # ant-like body
    bug_y_coords = [bug.y, bug.y - 2 * sin(bug.angle), bug.y - 4 * sin(bug.angle)]
    bug_plot = plot(bug_x_coords, bug_y_coords, 'ko')  # Plot the bug's current position
    #sr_plot = plot([bug.x, sr.x], [bug.y, sr.y], 'k')  # plot the antenna
    # sl_plot = plot([bug.x, sl.x], [bug.y, sl.y], 'k')     #plot the antenna
    food_plot = plot(foodx, foody, food_color)
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
plt.plot(foodx, foody, food_color)
axis([-100, 100, -100, 100])
title('Path')
show()
plt.plot(MB.x[0], MB.y[0])
plt.plot(foodx, foody, food_color)
axis([-100, 100, -100, 100])
title('Path')
show()