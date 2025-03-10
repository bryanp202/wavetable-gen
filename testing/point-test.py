from PlunSounds.Wavetable import Wavetable
import time

t = time.time()
wave = Wavetable('point-test')
def gener(a,b):
    y = a
    while True:
        yield y/b
        y *= a
        y %= b
values = gener(1394254,10321323)
for y in range(256):
    for i in range(64):
        wave.addpoint(i*32,next(values),'lin',2)
    wave.drawenv()
    wave.editwave('self.env1_fi(index)',minframe=1+y,maxframe=1+y)
wave.setdc()
wave.normalize('frame')
#wave.compress(0.5,.20,0.8,.20)
wave.exportwav('point-test-s.wav')
wave.exportpng()
dt = time.time()-t
print(dt)
wave.importwav('sin.wav','aux1')
print(wave.aux1_time)
wave.exportwav('sin2.wav','aux1')
