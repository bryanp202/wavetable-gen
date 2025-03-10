from PlunSounds.Wavetable import Wavetable

wave = Wavetable('frameimport-test',seed=3)
wave.importframe('test_frame.jpg')
wave.editdc()
wave.editphase('self.randi(index)*2*pi')
wave.editwave('1-2*((phase-0.2*x)%1)')
wave.exportwav('frameimport-test.wav')
