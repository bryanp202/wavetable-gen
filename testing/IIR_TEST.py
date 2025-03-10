from PlunSounds.Wavetable import Wavetable
from math import cos, pi

wave = Wavetable('iir-test')
wave.editwave(func='1-2*phase')
theta = 0.96*pi
function = "x+"
xcoef = (4*(cos(theta/2)**2),(2+4*cos(theta)),4*(cos(theta/2)**2),1)
ycoef = (-3.6*(cos(pi/2)**2),-0.8*(2+4*cos(pi)),-3.6*(cos(pi/2)**2),-.81)
for i,xk in enumerate(xcoef,1):
    if xk == 0:
        continue
    function += f"{xk}*self.main_fi_time(frame*255,(index-{i})%2048)+"
for i, yk in enumerate(ycoef,1):
    if yk == 0:
        continue
    function += f"{yk}*ybuff[int(frame*255),(index-{i})%2048]+"
print(function[:-1])
wave.editwave(func=function[:-1],yinit=0)
wave.editwave(func='x/(10+2*sqrt(5))',yinit=0)
print(list(wave.main_time[255]))
wave.editphase()
wave.setdc()
wave.normalize(mode="frame")
print(list(wave.main_time[255]))
wave.exportwav()
wave.exportpng()



#0.99995231666+0.009765470i
