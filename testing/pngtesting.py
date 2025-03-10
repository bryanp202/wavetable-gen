from PlunSounds.Wavetable import Wavetable
import time
from math import pi
import cmath, cv2


t= time.time()
wave = Wavetable('exportpngtest')
wave.editfreq('-1j/index')
#wave.editwave('sin(2*pi*phase+2*x*frame/255)')
#wave.editwave('1-(2*(phase + frame/255*x))%2')
wave.editphase('3*pi/2 + 2*pi*index/1024')
#wave.editphase()


wave.exportpng(mode='freq',colormode='color',shape=(1024,1024))
wave.exportpng(mode='time',colormode='color',shape=(1024,1024))

wave.exportwav()


wave2 = Wavetable('importtest')
wave2.importtable('exportpngtest-freq.png',mode='freq')
wave2.editdc()
wave2.exportwav()
dt = time.time() - t
print(dt)
