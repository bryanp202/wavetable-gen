from PlunSounds.Wavetable import Wavetable
from math import cos, pi

wave = Wavetable('iir-test',frames=256)
wave.editwave(func='1-2*phase')
for _ in range(1):
    wave.filter(z=[[1,0.01],[1,0.02],[1,0.03]],p='[[.8+0.192*frame/127,0.01],[.8+0.192*frame/127,0.02],[.8+0.192*frame/127,0.03]]',maxframe=127)
    wave.filter(z='[[0.9+0.1*frame/127,0.01],[0.9+0.1*frame/127,0.02],[0.9+0.1*frame/127,0.03]]',p=[[.992,0.01],[.992,0.02],[.992,0.03]],minframe=128)
#wave.filter(z=[[1,0.0009765625*3]],p=[[0.99,0.0009765625*3]])
wave.normalize(mode='frame')

"""
xcoef = [1, '2*cos(2*pi*frame/255)', 1.0]
function = f'{xcoef.pop(0)}*self.main_fi_time(frame,index)'
ycoef = ['2*cos(2*pi*frame/255)*0.99', 0.9801]
for i,xk in enumerate(xcoef,1):
    if not xk:
        continue
    function += f"+{xk}*self.main_fi_time(frame,(index-{i})%2048)"
for i, yk in enumerate(ycoef,1):
    if not yk:
        continue
    function += f"+-{yk}*ybuff[frame,(index-{i})%2048]"
print(function)
wave.editwave(func=function,yinit=0)
"""
#wave.editphase()
wave.exportwav(filename='iir-test2.wav')
wave.exportpng()

#0.99995231666+0.009765470i


####
"""
FIX IT SO THIS FILTER WORKKKSSSS

0.0 sec
Zero coefficients:
 [1.0, -1.859552971776503, 1.0]
Pole coefficients:
 [0.9999999999999999, -1.8783363351277804, 1.0203040506070808]
|(1.000000-1.859553*e^(-i*1*x)+1.000000*e^(-i*2*x))/(1.000000-1.878336*e^(-i*1*x)+1.020304*e^(-i*2*x))| from 0 to pi
"""
