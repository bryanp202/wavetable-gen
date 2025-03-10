from PlunSounds.Wavetable import Wavetable

table1 = Wavetable('image-testing',seed=123123)
table1.importframe('frame-drawing.png','time')
table1.editwave("sin(2*pi*phase + 10*x*frame/255)")
table1.editfreq("x - x*min((index-1)/(500-499*pow(frame/255,1/10)),1)")
#table1.editphase("self.randi(index)*pi*(1-frame/255)/4+x*frame/255*3/4")
#table1.normalize("frame")
table1.exportwav()
table1.exportpng(colormode='color',mode='freq',shape=(1024,1024))
