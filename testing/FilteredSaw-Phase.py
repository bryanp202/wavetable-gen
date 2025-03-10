from PlunSounds.Wavetable import Wavetable

wave = Wavetable(name='filtered-sawwave-phase')
wave.editfreq(func='-1j/index')
function = 'x+'
for i in range(1,40):
    function += f'self.main_fi_time(frame,(index-{i})%2048)+self.main_fi_time(frame,(index+{i})%2048)+'
wave.editwave(func=function[:-1])

wave.exportwav()
