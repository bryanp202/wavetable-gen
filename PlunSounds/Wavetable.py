import wave, array, struct, random, numpy, cv2
from cmath import rect, phase
from math import pi, sin, floor, cos, ceil, tan, asin, acos, atan, sqrt, log
from numpy.fft import rfft,irfft

#Max numFrames allowed
MAX_FRAMES = 256
#Max numChan allowed
MAX_CHANNELS = 1
#Maximum samplerate
MAX_SAMP_RATE = 128000
#Length of frame
FRAME_LEN = 2048
#Number if partials
NUM_PARTIALS = FRAME_LEN//2
#Time update position
TIME_UPDATE_INDEX = 0
FREQ_UPDATE_INDEX = 1
#Stores data type for different sampwidths
DATA_TYPE = {
    1: 'b',
    2: 'h',
    4: 'i'
    }
#Stores what buffers are currently supported
DATA_BUFFERS = ('main','aux1','aux2')
"""
Wavetable class
Useful for creating and editing wavetables, exporting and importing to .wav files
Const takes in a 'filename' str for starting from a existing .wav file or arguments for constructing a new .wav file
"""
class Wavetable:
    

    ######### Basic functionality functions
    """
    Creates wavetable object

    str name: name of the wavetable; is the default export name
    int frames: number of frames that the wavetable consideres active; must be in range [1,256]
    int samplerate: samplerate used only in exporting the wave, must be in range [1,128000]
    int width: width of each sample; acceptable widths are [1,2,4] ***Note width of 1 tends to cause most applications to not register the samples as signed integers
    int channels: number of channels output files will have; must be 1
    (int,float,str) seed: used as the seed for random number generators
    """
    def __init__(self, name='untitled', frames=256, samplerate=44100, width=2, channels=1, seed=None):
        global MAX_CHANNELS, MAX_FRAMES, FRAME_LEN, NUM_PARTIALS, FRAME_RNG_OFFSET, MAX_SAMP_RATE
        if not(width == 1 or width == 2 or width == 4):
            raise RuntimeError('Invalid sample byte width')
        if channels > MAX_CHANNELS or channels < 0:
            raise RuntimeError('Invalid amount of channels')
        if frames > MAX_FRAMES or frames <= 0:
            raise RuntimeError('Invalid amound of frames')
        if samplerate > MAX_SAMP_RATE or samplerate < 0:
            raise RuntimeError('Invalid framerate')
        if not (name or type(name) is str) or '.' in name:
            raise RuntimeError('Invalid name')

        self.name = name
        self.sampwid = width
        self.amp = 2**(self.sampwid*8-1)-1
        self.numchan = channels
        self.samprate = samplerate
        self.numframes = frames
        
        #For each buffer, create a regulator bool and list of frames of sound data for the time and freq domain
        self.main_update, self.main_time, self.main_freq = ([True, True],numpy.zeros(shape=(MAX_FRAMES,FRAME_LEN)),numpy.zeros(shape=(MAX_FRAMES,NUM_PARTIALS+1),dtype=complex))
        self.aux1_update, self.aux1_time, self.aux1_freq = ([True, True],numpy.zeros(shape=(MAX_FRAMES,FRAME_LEN)),numpy.zeros(shape=(MAX_FRAMES,NUM_PARTIALS+1),dtype=complex))
        self.aux2_update, self.aux2_time, self.aux2_freq = ([True, True],numpy.zeros(shape=(MAX_FRAMES,FRAME_LEN)),numpy.zeros(shape=(MAX_FRAMES,NUM_PARTIALS+1),dtype=complex))

        #Create a guide envelope for use in functions
        self.env1 = numpy.zeros(shape=(FRAME_LEN))
        #Create curvepoint list for each guide envelope
        self.env1_points = list()

        #Create random generator and random buffers
        self.rng = random.Random(seed)
        self.rngf = [self.rng.random() * 2 - 1 for f in range(MAX_FRAMES)]
        self.rngi =  [self.rng.random() * 2 - 1 for i in range(FRAME_LEN)]

    """
    Returns string containing import details about self
    """
    def __str__(self):
        return '[Name: %s, Frames: %i, Channels: %i, Sample Rate: %i, Sample Width: %i bytes]'%(self.name,self.numframes,self.numchan,self.samprate,self.sampwid)

    """
    Spectral Domain Processing helper functions
    Call fft when main changes
    Call ifft when main_spect changes
    """
    def fft(self, target='main'):
        global FREQ_UPDATE_INDEX
        
        timedata = self._get_target_time(target)
        freqdata = self._get_target_freq(target)
        update = self._get_target_update(target)
        
        for frame in range(self.numframes):
            freqdata[frame] = rfft(timedata[frame])
        update[FREQ_UPDATE_INDEX] = True

    def ifft(self, target='main'):
        global TIME_UPDATE_INDEX
        
        timedata = self._get_target_time(target)
        freqdata = self._get_target_freq(target)
        update = self._get_target_update(target)
        
        for frame in range(self.numframes):
            timedata[frame] = irfft(freqdata[frame])

        update[TIME_UPDATE_INDEX] = True

    """
    Exports target buffer data to location 'filename' defaults to self.name-edited.wav
    target defaults to 'main'
    """
    def exportwav(self, filename=None, target='main'):
        global TIME_UPDATE_INDEX, FRAME_LEN, DATA_TYPE

        timedata = self._get_target_time(target)
        update = self._get_target_update(target)
        
        if not update[TIME_UPDATE_INDEX]:
            self.ifft(target)
        
        self.normalize('total',target)

        #Deal with NaNs
        timedata[:] = numpy.nan_to_num(timedata,nan=0.0)

        if not filename:
            filename = '%s.wav' % self.name
        
        with wave.open(filename,'wb') as fi:
            fi.setparams((self.numchan,self.sampwid,self.samprate,FRAME_LEN*self.numframes,'NONE','not compressed'))
            #Process Data
            rawdata = array.array(DATA_TYPE[self.sampwid],[0 for i in range(self.numframes*FRAME_LEN)])
            timedata = (timedata * self.amp).astype(int)
            for frame in range(self.numframes):
                for point in range(FRAME_LEN):
                    rawdata[frame*FRAME_LEN+point] = timedata[frame,point]

            fi.writeframesraw(rawdata)
            print(f"Successfully exported buffer: '{target}', from {self} to file: {filename}")

    """
    Imports current wave from location 'filename' into buffer 'target'.
    Target defaults to 'main'
    """
    def importwav(self, filename, target='main'):
        global MAX_CHANNELS, MAX_FRAMES, FRAME_LEN, DATA_TYPE, FREQ_UPDATE_INDEX

        timedata = self._get_target_time(target)
        update = self._get_target_update(target)
        
        with wave.open(filename, 'rb') as fi:
            #Self attributes
            if target == 'main':
                self.sampwid = fi.getsampwidth()
                self.amp = 2**(self.sampwid*8-1)-1
                self.numchan = fi.getnchannels()
                self.samprate = fi.getframerate()
                self.numframes = min(fi.getnframes() // FRAME_LEN, MAX_FRAMES)
                #Locals for unpacking
                sampwid = self.sampwid
                amp = self.amp
                numchan = self.numchan                
            #If not main, do not update object attributes
            else:
                sampwid = fi.getsampwidth()
                amp = 2**(sampwid*8-1)-1
                numchan = fi.getnchannels()
            #Unpack data
            if numchan > MAX_CHANNELS:
                print('Imported .wav file has more than MAX_CHANNELS: %i channels; all other channels but the first will be ignored' % MAX_CHANNELS)
            numframes = min(fi.getnframes() // FRAME_LEN, MAX_FRAMES)
            unpackFormat = '<%i%s' % (FRAME_LEN*self.numchan, DATA_TYPE[self.sampwid])
            for frame in range(self.numframes):
                timedata[frame] = struct.unpack(unpackFormat, fi.readframes(FRAME_LEN))[::self.numchan]
            for frame in range(self.numframes, MAX_FRAMES):
                timedata[frame].fill(0)
            self.numchan = 1
            print(f"Successfully imported {self} into buffer '{target}' from file:", filename)

            self._normalize_init(target, amp)
            update[FREQ_UPDATE_INDEX] = False

    """
    Imports a png image as a single frame to targeted buffer
    
    farget defaults to 'main': which buffer to add input the frame
    minframe defaults to 1: First frame to edit
    maxframe defautls to 0: Last frame to edit
    min and max frame create bounds [minframe,maxframe]

    ***NOT guaranteed that the target buffer output will be bound for-1 to 1 for mode='freq'; could effect distortion effects***
    """
    def importframe(self, filename, mode='time', minframe=1, maxframe=0, target='main'):
        global TIME_UPDATE_INDEX, FREQ_UPDATE_INDEX, FRAME_LEN, NUM_PARTIALS, MAX_FRAMES
        #Extract data from image
        rawdata = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
        rawdata = numpy.flip(rawdata,axis=0)
        height, width = rawdata.shape
        height -= 1
        width -= 1
        framedata = numpy.argmin(rawdata,axis=0)
        #Get update pointer
        update = self._get_target_update(target)
        #Select mode and set up for that mode
        if mode == 'time':
            bufferdata = self._get_target_time(target)
            updateset = FREQ_UPDATE_INDEX
            length = FRAME_LEN
            framedata = (framedata*2 - height)/height
            overlayedindexes = numpy.array([x*length/width for x in range(0,width+1)])
            addstart = 0
            if not update[TIME_UPDATE_INDEX]:
                self.ifft(target)
        elif mode == 'freq':
            bufferdata = self._get_target_freq(target)
            updateset = TIME_UPDATE_INDEX
            length = NUM_PARTIALS
            framedata = (pow(length,framedata/height)-1)/(length-1)
            framedata = framedata.astype(complex)*-1j*FRAME_LEN
            overlayedindexes = numpy.array([pow(length,x/(width)) - 1 for x in range(0,width+1)])
            addstart = 1
            if not update[FREQ_UPDATE_INDEX]:
                self.fft(target)
        #Check minframe and maxframe
        minframe, maxframe = self.frameRangeCheck(minframe,maxframe,self.numframes)
        #interpolate data
        indexes = numpy.array(list(range(0,length)))
        framedata = numpy.interp(indexes,overlayedindexes,framedata)
        bufferdata[minframe:maxframe,addstart:] = framedata[:]

        update[updateset] = False
        print(f"Successfully imported frame from image into buffer '{target}' ranging frames [{minframe+1}-{maxframe}] from file: {filename}")

    """
    Used to import any png,jpg,etc into a targeted wavetable buffer
    Can either interpret data as points in time or as values for partials

    filename: target location to extract img data
    mode: either 'time' or 'freq'
        time: how pure a pixel is determines what value [-1,1] each index in the targets time buffer is
        freq: magnitude is similar to time mode, but effects the targets freq buffer instead
    minframe: first frame to import extracted data
    maxframe: last frame to import extracted data
    target: targeted buffer
    withphase: does nothing in time mode, but in freq mode will cause the location of a color on the color wheel to determine a partials phase
    """
    def importtable(self, filename, mode='time', minframe=1, maxframe=0, target='main', withphase=True):
        global TIME_UPDATE_INDEX, FREQ_UPDATE_INDEX, FRAME_LEN, NUM_PARTIALS
        #Get update pointer
        update = self._get_target_update(target)
        #Check minframe and maxframe
        minframe, maxframe = self.frameRangeCheck(minframe,maxframe,self.numframes)
        #Select mode and set up for that mode
        if mode == 'time':
            bufferdata = self._get_target_time(target)
            updateset = FREQ_UPDATE_INDEX
            width = FRAME_LEN
            height = maxframe-minframe
            #Get image data and reshape it
            img = self._getimgdata(filename, height, width)
            editedx = img - numpy.min(img, axis=2).reshape(height,width,1)
            mag = (numpy.max(editedx,axis=2) - 128)/255
            #Update buffer if needed
            if not update[TIME_UPDATE_INDEX]:
                self.ifft(target)
            bufferdata[minframe:maxframe] = mag[:]
        elif mode == 'freq':
            bufferdata = self._get_target_freq(target)
            updateset = TIME_UPDATE_INDEX
            width = NUM_PARTIALS
            height = maxframe-minframe
            #Used to ignore DC element
            addstart = 1
            #Get image data and reshape it
            img = self._getimgdata(filename, height, width)
            editedx = img - numpy.min(img, axis=2).reshape(height,width,1)
            mag = NUM_PARTIALS/255*numpy.max(editedx,axis=2)
            #Update buffer if needed
            if not update[FREQ_UPDATE_INDEX]:
                self.fft(target)
            #Check if color should be interpreted as phase of each partial
            if withphase:
                phasereshape = editedx.reshape(width*height,3)
                phases = numpy.array(list(map(self._getphase,phasereshape.astype(int)))).reshape(height,width)
                for frame in range(maxframe-minframe):
                    phasedata = map(rect,mag[frame],phases[frame]*pi)
                    bufferdata[frame+minframe,addstart:] = numpy.array(list(phasedata), dtype=complex)
            else:
                bufferdata[minframe:maxframe,addstart:] = mag*-1j
        update[updateset] = False
        print(f"Successfully imported wavetable from image into buffer '{target}' from file: {filename}")

    @staticmethod
    def _getphase(line):
            maxloc = numpy.argmax(line)
            maxval = max(line[maxloc],1)
            return ((2-maxloc)*2/3 + max(line[(maxloc-1)%3]/(maxval*2), -line[(maxloc+1)%3]/(maxval*6), key=abs))%2
    @staticmethod
    def _getimgdata(filename, height, width):
        img = cv2.imread(filename)
        imgwidth, imgheight, imgdepth = img.shape
        #Choose appropriate interp algo for shrinking or zooming
        if (width + height - imgheight - imgwidth) < 0:
            #Shrink
            return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA).astype(int)
        else:
            #Zoom
            return cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR).astype(int)
        
    ######### Functions for users to manipulate data
    """
    Normalizes self.main based of the following modes:
        'init': assumes data has just been imported; normalizes everything by self.amp
        'total': normalizes based off the highest abs value in the entire main buffer
        'frame': normalizes each frame based on the highest value in that frame

        minframe 1 is equal to first frame, minframe -1 is equal to second to last frame
        maxframe 0 is equal to last frame, last frame -1 is equal to second to last frame

        func should be a algabric function of db to change each frame by
        NOTE: 'frame' variable used in functions ranges from 0 to 1 for frames between minframe and maxframe
        eg: '0' is equal to setting the function to 0db, or bound to amplitudes -1 to 1
        eg: '0 + frame' causes the function to go from 0db on the first frame to NUMFRAMES db on the last 
    """
    def normalize(self, mode='total', target='main', minframe=1, maxframe=0, func = '0'):
        global FREQ_UPDATE_INDEX, TIME_UPDATE_INDEX

        dtimedata = self._get_target_time(target)
        update = self._get_target_update(target)
        #Check minframe and maxframe
        minframe, maxframe = self.frameRangeCheck(minframe,maxframe,self.numframes)

        if not update[TIME_UPDATE_INDEX]:
            self.ifft(target)
        
        if not func == '0':
            dbFunc = eval('lambda frame: 10**((%s)/20)'%func)
        else:
            dbFunc = eval('lambda frame: 1')
        
        if mode == 'total':
            self._normalize_total(target, minframe, maxframe, dbFunc)
        elif mode == 'frame':
            self._normalize_frame(target, minframe, maxframe, dbFunc)
        else:
            raise RuntimeError('Invalid mode: %s'%mode)

        update[FREQ_UPDATE_INDEX] = False
        
    """
    Normalize helper functions
    """
    def _normalize_init(self, target, amp):
        timedata = self._get_target_time(target)
        timedata /= amp
                    
    def _normalize_total(self, target, minframe, maxframe, dbFunc):
        timedata = self._get_target_time(target)
        highestPoint = max(timedata.max(), abs(timedata.min()))
        if not highestPoint == 0:
            for frame in range(minframe,maxframe):
                timedata[frame] *= dbFunc(frame) / highestPoint
                    
    def _normalize_frame(self, target, minframe, maxframe, dbFunc):
        timedata = self._get_target_time(target)
        highestPoints = (max(timedata[x].max(),abs(timedata[x].min())) for x in range(minframe,maxframe))
        for frame, highestPoint in enumerate(highestPoints,minframe):
            if not highestPoint == 0:
                timedata[frame] *= dbFunc(frame) / highestPoint
    
    """
    Apply a filter with zeros and poles at designated locations
    """
    def filter(self, z=[], p=[], target='main', minframe=1, maxframe=0):
        global MAX_FRAMES, FRAME_LEN, TIME_UPDATE_INDEX, FREQ_UPDATE_INDEX

        timedata = self._get_target_time(target)
        update = self._get_target_update(target)

        #Check minframe and maxframe
        minframe, maxframe = self.frameRangeCheck(minframe,maxframe,self.numframes)
        #Check if target spectrum data is updated and update it if needed
        if not update[TIME_UPDATE_INDEX]:
            self.ifft(target)

        zeroFunc = 0
        poleFunc = 0
        if type(z) is str:
            zeroFunc = eval(f'lambda frame: ({z})')
        else:
            filtZeros = self.expandFactors(z)
        if type(p) is str:
            poleFunc = eval(f'lambda frame: ({p})')
        else:
            filtPoles = self.expandFactors(p)[1:]

        tempdata = numpy.zeros(shape=(self.numframes,FRAME_LEN))
        
        for frame in range(minframe,maxframe):
            if zeroFunc:
                filtZeros = self.expandFactors(zeroFunc(frame))
            if poleFunc:
                filtPoles = self.expandFactors(poleFunc(frame))[1:]
            for index in range(FRAME_LEN*2):
                currIndex = 0
                for i,zero in enumerate(filtZeros):
                    currIndex += zero*timedata[frame,(index-i)%FRAME_LEN]
                for i,pole in enumerate(filtPoles,1):
                    currIndex += -pole*tempdata[frame,(index-i)%FRAME_LEN]
                tempdata[frame,(index)%FRAME_LEN] = currIndex
        #Sets update bool
        timedata[minframe:maxframe] = tempdata[minframe:maxframe]
        update[FREQ_UPDATE_INDEX] = False

    """
    Edit the frequency domain information of target buffer

    target is the targeted buffer defaults to 'main'

    minframe and maxframe are an inclusive range
    if either minframe or maxframe are negative, wraps around
    maxframe = 0 is equal to last frame

    func should be like form 'sin(2*pi*phase)'
    can use all math functions and the variables:
    y: the last number calculated or the yinit if at minindex
    x: the prior value of the buffer before applying the formula
    phase: value 0 to 1 for current index being edited
    index: the value from 0 to FRAME_LEN of the current index being edited
    frame: current frame

    Can use all globals in equation

    Can use get functions like self.main_fi_time(frame, index)

    minindex and maxindex control which indeces of each frame are edited
    makes the range [minindex,maxindex)
    """
    def editwave(self, func='0', target='main', minframe=1, maxframe=0, yinit=0.0, minindex=0, maxindex=None):
        global MAX_FRAMES, FRAME_LEN, TIME_UPDATE_INDEX, FREQ_UPDATE_INDEX

        timedata = self._get_target_time(target)
        update = self._get_target_update('all')

        #Set maxindex default
        if maxindex == None:
            maxindex = FRAME_LEN
        #Check minframe and maxframe
        minframe, maxframe = self.frameRangeCheck(minframe,maxframe,self.numframes)

        #Check if target spectrum data is updated and update it if needed
        if not update['main'][TIME_UPDATE_INDEX]:
            self.ifft(target)
        if 'aux1' in func and not update['aux1'][TIME_UPDATE_INDEX]:
            self.ifft('aux1')
        if 'aux2' in func and not update['aux2'][TIME_UPDATE_INDEX]:
            self.ifft('aux2')
        
        timeFunc = eval(f'lambda self, x, y, frame, phase, index, ybuff: ({func})')#        
        
        tempdata = numpy.zeros(shape=(self.numframes,FRAME_LEN))
        
        for frame in range(minframe,maxframe):
            y = yinit
            for index in range(minindex, maxindex):
                x = timedata[frame,index]
                y = timeFunc(self,x,y,frame,index/FRAME_LEN,index,tempdata)
                tempdata[frame,index] = y
        #Sets update bool
        timedata[minframe:maxframe,minindex:maxindex] = tempdata[minframe:maxframe,minindex:maxindex]
        update[target][FREQ_UPDATE_INDEX] = False

    """
    Edit the frequency domain information of target buffer

    target is the targeted buffer defaults to 'main'

    refers to the index of the partial between 1 and 1024 (dc is handled by editdc)
    Minindex must be between 1 and 1024
    Maxindex must be between 1 and 1024
    Both are inclusive

    minframe and maxframe are an inclusive range
    if either minframe or maxframe are negative, wraps around

    func should be like form '1/index'
    can use all math functions and the variables:
    y: the last number calculated or the yinit if at minindex
    x: the prior value of the buffer before applying the formula
    index: the value of the current index being edited
    frame: current frame

    Can use get functions like main_fi_freq(frame, index)
    """
    def editfreq(self, func='0', target='main', minindex=1, maxindex=None, minframe=1, maxframe=0, yinit=0.0):
        global MAX_FRAMES, FRAME_LEN, TIME_UPDATE_INDEX, FREQ_UPDATE_INDEX, NUM_PARTIALS

        freqdata = self._get_target_freq(target)
        update = self._get_target_update('all')

        if maxindex == None:
            maxindex = NUM_PARTIALS
        if minindex <= 0:
            raise RuntimeError('minimum index must be greater or equal to 1 (fundamental)')
        #Check minframe and maxframe
        minframe, maxframe = self.frameRangeCheck(minframe,maxframe,self.numframes)
        #Make maxindex inclusive
        maxindex += 1

        #Check if target spectrum data is updated and update it if needed
        if not update['main'][FREQ_UPDATE_INDEX]:
            self.fft(target)
        if 'aux1' in func and not update['aux1'][FREQ_UPDATE_INDEX]:
            self.fft('aux1')
        if 'aux2' in func and not update['aux2'][FREQ_UPDATE_INDEX]:
            self.fft('aux2')

        spectFunc = eval(f'lambda self, x, y, frame, index: ({func})')#        
        tempdata = numpy.zeros(shape=(self.numframes,NUM_PARTIALS+1),dtype=complex)
        
        for frame in range(minframe, maxframe):
            y = yinit
            freqdata[frame] /= FRAME_LEN
            for index in range(minindex, maxindex):
                x = freqdata[frame,index]
                y = spectFunc(self,x,y,frame,index)
                tempdata[frame,index] = y
            tempdata[frame] *= FRAME_LEN
        #Sets update bool
        freqdata[minframe:maxframe,minindex:maxindex] = tempdata[minframe:maxframe,minindex:maxindex]
        update[target][TIME_UPDATE_INDEX] = False

    """
    Sets value of dc of set

    Both are inclusive

    func should be like form 'sin(2*pi*frame/FRAME_LEN)'
    can use all math functions and the variables:
    frame: current frame

    Can use all globals in equation

    Can use get functions like self.main_fi_time(..., index)

    [minindex,maxindex) control which indeces are edited

    buffers can be accessed in equations with self.env1
    """
    def editdc(self, func='0', target='main', minframe=1, maxframe=0):
        global FRAME_LEN, TIME_UPDATE_INDEX, NUM_PARTIALS

        freqdata = self._get_target_freq(target)
        update = self._get_target_update(target)

        #Check if target spectrum data is updated and update it if needed
        if not update[FREQ_UPDATE_INDEX]:
            self.fft(target)
        #Check minframe and maxframe
        minframe, maxframe = self.frameRangeCheck(minframe,maxframe,self.numframes)
        
        spectFunc = eval('lambda frame: (%s)'%func)

        for frame in range(minframe, maxframe):
            freqdata[frame,0] = spectFunc(frame)*FRAME_LEN

        #Sets update bool
        update[TIME_UPDATE_INDEX] = False

    """
    Sets the phase of target buffer
    Goes from index 1 - 1024

    minindex and maxindex are inclusive

    minframe and maxframe wrap around
    maxframe of 0 is equal to last frame

    func defaults to -pi/2, aka everything is a sin wave
    yinit is the starting value of equation buffer
    Can use global variables
    x: phase before editing
    y: last calculated result
    frame: current frame
    """
    #FIX SO ROTATING A WAVE IS MORE PRECISE
    def editphase(self, func='-pi/2', target='main', minindex=1, maxindex=None, minframe=1, maxframe=0,yinit=0):
        global FREQ_UPDATE_INDEX, TIME_UPDATE_INDEX, NUM_PARTIALS

        freqdata = self._get_target_freq(target)
        update = self._get_target_update(target)
        #Check minframe and maxframe
        minframe, maxframe = self.frameRangeCheck(minframe,maxframe,self.numframes)
        if maxindex == None:
            maxindex = NUM_PARTIALS
        #Max maxindex inclusive
        maxindex += 1

        #Check if target spectrum data is updated and update it if needed
        if not update[FREQ_UPDATE_INDEX]:
            self.fft(target)

        phaseFunc = eval('lambda self, x, y, frame, index: (%s)'%func)

        for frame in range(minframe, maxframe):
            y = yinit
            for index in range(minindex, maxindex):
                x = freqdata[frame,index]
                y = phaseFunc(self, phase(x),y,frame,index) % (2*pi)
                freqdata[frame,index] = rect(abs(x), y)

        #Sets update bool
        update[TIME_UPDATE_INDEX] = False

    """
    ***ASSUMES THE DATA IN THE BUFFER IS NORMALIZED BETWEEN [-1,1]
        *It will still work if not bounded, but threshold values will potentially be misaligned
    Compresses a targeted buffer
    Acts like a traditional digital compressor:
        anything below the minthresh is effected by the ratio minratio
        anything above the maxthresh is effected by the ratio maxratio
        No soft knee at this time

    minframe and maxframe wrap around
    maxframe of 0 is equal to last frame
    """
    #STILL PROBABLY SOME BUGS WARNING *******
    def compress(self, maxthresh, maxratio, minthresh=0, minratio=1, target='main', minframe=1, maxframe=0):
        global FRAME_LEN, TIME_UPDATE_INDEX, FREQ_UPDATE_INDEX
        
        timedata = self._get_target_time(target)
        update = self._get_target_update(target)
        #Check minframe and maxframe
        minframe, maxframe = self.frameRangeCheck(minframe,maxframe,self.numframes)

        #Check if target spectrum data is updated and update it if needed
        if not update[TIME_UPDATE_INDEX]:
            self.ifft(target)

        for frame in range(minframe, maxframe):
            frameAbsMax = 0
            for index in range(0, FRAME_LEN):
                x = timedata[frame,index]
                if x >= 0:
                    if x < minthresh:
                        timedata[frame,index] = max(x/minratio - minthresh*(1/minratio-1), 0)
                    elif x > maxthresh:
                        timedata[frame,index] = maxthresh + (x-maxthresh)/maxratio
                else:
                    if x > -minthresh:
                        timedata[frame,index] = max(x/minratio + minthresh*(1/minratio-1), 0)
                    elif x < -maxthresh:
                        timedata[frame,index] = (x+maxthresh)/maxratio - maxthresh
            frameAbsMax = max(timedata[frame])
            timedata[frame] *= frameAbsMax/(maxthresh + (frameAbsMax-maxthresh)/maxratio)

        #Sets update bool
        update[FREQ_UPDATE_INDEX] = False

    #ENV RELATED FUNCTIONS
    #Envelopes are one frame, 0-2048 index buffer that can be drawn with points
    #Useful for windowing effects or creating geometric shapes quickly
    """
    Edits target env curve, defaults to 1 (env1)

    func defaults to 
    yinit is the starting value of equation buffer
    Can use global variables
    x: phase before editing
    y: last calculated result
    yinit = z^-1
    ybuff = z^-n ***Note will cause blurring 

    [minindex,maxindex)
    """
    def editenv(self, func='0', target=1, yinit=0.0, minindex=0, maxindex=None):
        global MAX_FRAMES, FRAME_LEN, TIME_UPDATE_INDEX, FREQ_UPDATE_INDEX

        env = self._get_target_env(target)

        #Set maxindex default
        if maxindex == None:
            maxindex = FRAME_LEN
        
        timeFunc = eval(f'lambda self, x, y, phase, index, ybuff: ({func})')        
        
        tempdata = numpy.zeros(shape=(FRAME_LEN))
        
        y = yinit
        for index in range(minindex, maxindex):
            x = env[index]
            y = timeFunc(self,x,y,index/FRAME_LEN,index,tempdata)
            tempdata[index] = y
        #update envelope
        env[minindex:maxindex] = tempdata[minindex:maxindex]

    """
    Reverves a target env curve
    """
    def flipenv(self, target=1):
        env = self._get_target_env(target)
        env[:] = numpy.flip(env)

    """
    Adds a point to the target envelope

    time = timeindex to add the point
    value = value to add at designated time (usually -1<=value<=1)
    eqform = what shape to connect to this point from the previous point
        [lin, exp, invexp(aka sudolog), s]
    eqvalue = augments the desired eqform if possible (could make exp steeper)
        defaults to 0, eqvalue<0 is usually less steep, eqvalue>0 is usually more intense

    Once all desired points are saved, call self.drawenv(target) to connect the points
    """
    def addpoint(self, time, value, eqform='lin', eqvalue=0,target=1):
        envpoints = self._get_target_env_points(target)
        envpoints.append(((time,value),(eqform,eqvalue)))

    """
    Automatically connects all points in 
    """
    def drawenv(self, target=1):
        global FRAME_LEN
        #Stores func pointer for different curve point connecting functions
        CURVE_FUNC = {
            'lin': self._lin,
            'exp': self._exp,
            'invexp': self._invexp,
            's': self._scurve
        }
        env = self._get_target_env(target)
        envpoints = self._get_target_env_points(target)
        envpoints.sort(key=lambda x: x[0][0])

        length = len(envpoints)
        for i in range(length):
            minindex, startvalue = envpoints[i][0]
            eqform, eqvalue = envpoints[i][1]
            maxindex, endvalue = envpoints[(i+1)%length][0]
            if maxindex == minindex:
                maxindex += FRAME_LEN
            elif maxindex < minindex:
                maxindex += FRAME_LEN
            func = CURVE_FUNC[eqform](minindex,maxindex,startvalue,endvalue,eqvalue)
            for index, value in enumerate(func, minindex):
                env[index%FRAME_LEN] = value
                
        #update envelope points
        envpoints[:] = list()
    
    """
    Generators for different curve shapes

    minindex, maxindex: range of indexes the function is bound to
    startvalue, endvalue: range of y values the function is bound to

    eqvalue: alters characterstics of certain func types
    """
    #eqvalue does nothing
    @staticmethod
    def _lin(minindex,maxindex,startvalue,endvalue,eqvalue):
        length = maxindex-minindex
        slope = (endvalue - startvalue) / length
        value = startvalue
        for i in range(0,length):
            yield value + i*slope
    #eqvalue shifts the exponent value, which defaults to 2
    #exponent value cannot be less than 0.01
    @staticmethod
    def _exp(minindex,maxindex,startvalue,endvalue,eqvalue):
        length = maxindex - minindex
        offset = endvalue - startvalue
        eqvalue = max(2+eqvalue,0.01)
        for i in range(0,length):
            yield pow(i/length,eqvalue)*offset + startvalue
    #eqvalue shifts the exponent value, which defaults to 2
    #exponent value cannot be less than 0.01
    #Sudo log generator, inverted exponential curve
    @staticmethod
    def _invexp(minindex,maxindex,startvalue,endvalue,eqvalue):
        length = maxindex - minindex
        offset = endvalue - startvalue
        eqvalue = max(2+eqvalue,0.01)
        for i in range(length,0,-1):
            yield endvalue - pow(i/length,eqvalue)*offset
    #eqvalue changes how sharply the s-curve curves
    @staticmethod
    def _scurve(minindex,maxindex,startvalue,endvalue,eqvalue):
        length = maxindex - minindex
        offset = startvalue - endvalue
        a = 10
        b = max(1,eqvalue)
        ab_plus = pow(a,b) + 1
        ab_minus = ab_plus - 2
        for i in range(0,length):
            yield ((ab_plus) / (ab_minus * (1 + pow(a, 2*b*i/length - b))) - 1/ab_minus)*offset + endvalue 

    """
    Exports target spectrum data as png

    valid sizes are [1,2,4] *** dependent on MAX_FRAMES and FRAME_LEN's ratio
    """
    def exportpng(self, filename=None, target='main', mode='time', colormode='greyscale', shape=(256,256)):
        global TIME_UPDATE_INDEX, MAX_FRAMES, FRAME_LEN
        #Normalize data for export
        self.normalize('total',target)
        #Get update buffer
        update = self._get_target_update(target)
        
        if filename == None:
            filename = f"{self.name}"
        if mode == 'time':
            filename += '-time.png'
            if not update[TIME_UPDATE_INDEX]:
                self.ifft(target)
            dataout = self._get_target_time(target)
            width = FRAME_LEN
            if colormode == 'greyscale':
                dataout = numpy.nan_to_num((dataout/2+0.5)*255)
            if colormode == 'color':
                dataout = numpy.nan_to_num(list(map(self._getampcolor,dataout.flatten()))).reshape(256,width,3)
        if mode == 'freq':
            filename += '-freq.png'
            if not update[FREQ_UPDATE_INDEX]:
                self.fft(target)
            dataout = self._get_target_freq(target)[:,1:]
            width = NUM_PARTIALS
            if colormode == 'greyscale':
                dataout = numpy.nan_to_num(abs(dataout)*255)
            if colormode == 'color':
                dataout = numpy.nan_to_num(list(map(self._getphasecolor,dataout.flatten()))).reshape(256,width,3)
            #Rescale partials
            maxval = abs(max(dataout.max(),dataout.min(),key=abs))
            if maxval == 0:
                maxval = 1
            dataout = (numpy.array(list(map(log,1+1023/maxval*dataout.flatten())))).reshape(256,width,3)*255/log(1024)
        #Choose appropriate interp algo for shrinking or zooming
        dataout = dataout.astype(numpy.uint8)
        if (shape[0] + shape[1] - width - 256) < 0:
            #Shrink
            dataout = cv2.resize(dataout, shape, interpolation=cv2.INTER_AREA)
        else:
            #Zoom
            dataout = cv2.resize(dataout, shape, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(filename,dataout)
        print(f"Successfully exported buffer: '{target}', from {self} to file: {filename}")
    """
    Used to get the color for a partials phase
    """
    @staticmethod
    def _getphasecolor(complexamp):
        amp = int(255*abs(complexamp))
        phi = (phase(complexamp) / pi)%2
        currval = phi-1/3
        if currval <= 0:
            return (0,int((1+3*currval)*amp),amp)
        currval -= 1/3
        if currval <= 0:
            return (0,amp,int(-currval*3*amp))
        currval -= 1/3
        if currval <= 0:
            return (int((1+3*currval)*amp),amp,0)
        currval -= 1/3
        if currval <= 0:
            return (amp,int(-currval*3*amp),0)
        currval -= 1/3
        if currval <= 0:
            return (amp,0,int((1+3*currval)*amp))
        currval -= 1/3
        return (int(-currval*3*amp),0,amp)
    """
    Returns color for time mode exports in color mode
    """
    @staticmethod
    def _getampcolor(amp):
        if amp > 0:
            return (int(amp*255),0,0)
        else:
            return (0,0,int(-amp*255))
    
    
    ######### Functions for changing or using basic attributes
    #Sets name of wavetable
    def setname(self, name):
        self.name = name
    #Sets width of samples and adjustes max amplitude
    def setwidth(self, width):
        self.sampwid = width
        self.amp = 2**(self.sampwid*8-1)-1
    #Edit number of frames for editing and exporting
    def setframes(self, numframes):
        global MAX_FRAMES, DATA_BUFFERS
        self.numframes = numframes
        for buff in DATA_BUFFERS:
            timedata = self._get_target_time(buff)
            freqdata = self._get_target_freq(buff)
            for frame in range(self.numframes, MAX_FRAMES):
                timedata[frame].fill(0)
                freqdata[frame].fill(0)
    #Set sample rate, does nothing for wavetables but might change regular playback
    def setsamprate(self, samprate):
        self.samprate = samprate

    """
    Clears targeted buffers
    """
    def clear(self, target='main'):
        global MAX_FRAMES, FRAME_LEN, NUM_PARTIALS
        timedata = self._get_target_time(target)
        freqdata = self._get_target_freq(target)
        update = self._get_target_update(target)
        timedata[:] = numpy.zeros(shape=(MAX_FRAMES,FRAME_LEN))
        freqdata[:] = numpy.zeros(shape=(MAX_FRAMES,NUM_PARTIALS+1),dtype=complex)
        update[:] = [True,True]

    """
    Clears targeted env and env points
    """
    def clear_env(self, target=1):
        global FRAME_LEN
        env = self._get_target_env(target)
        envpoints = self._get_target_env_points(target)
        env[:] = numpy.zeros(shape=(FRAME_LEN))
        envpoints[:] = list()

    #Functions for users to get data
    """ 
    Used by users to access certain indexes of a buffer from the lambda function
    Takes in buffer target, frame, and index

    Index and frame will be rounded to nearest int

    Works for time domain

    Accessed by self.[buffer-name]_fi_time(frame, index)
    """
    def main_fi_time(self, frame, index):
        return self.main_time[int(frame),int(index)]
    def aux1_fi_time(self, frame, index):
        return self.aux1_time[int(frame),int(index)]
    def aux2_fi_time(self, frame, index):
        return self.aux2_time[int(frame),int(index)]
    """ 
    Used by users to access certain indexes of a buffer from the lambda function
    Takes in buffer target, frame, and index

    Index and frame will be rounded to nearest int

    Works for freq domain

    Accessed by self.[buffer-name]_fi_freq(frame, index)
    """
    def main_fi_freq(self, frame, index):
        global FRAME_LEN
        return self.main_freq[int(frame),int(index)]/FRAME_LEN
    def aux1_fi_freq(self, frame, index):
        global FRAME_LEN
        return self.aux1_freq[int(frame),int(index)]/FRAME_LEN
    def aux2_fi_freq(self, frame, index):
        global FRAME_LEN
        return self.aux2_freq[int(frame),int(index)]/FRAME_LEN

    """
    Used by users to access certain indexes of a env curve buffer from within a lambda function
    Takes in buffer target and index

    Index will be rounded to nearest int, so input can be any numerical type

    Accessed by self.[env-name]_fi(index)   
    """
    def env1_fi(self, index):
        return self.env1[int(index)]

    """
    Returns random number
    """
    def rand(self):
        return self.rng.random() * 2 - 1

    """
    Returns random number
    Value is dependent on inputted frame
    Same for all indexes
    """
    def randf(self, frame):
        return self.rngf[int(frame)]

    """
    Returns random number
    Value is dependent on inputted sample
    Same for all frames
    """
    def randi(self, index):
        return self.rngi[int(index)]

    ######### Helper functions
    """
    Edits the user input string to something that eval() can read

    func: string to be editted

    returns reconfigured func str
    """
    def configfunc(self, func):
        pass

    
    """
    Return pointer to the correct buffer to edit
    """
    def _get_target_time(self, target):
        if target == 'main':
            return self.main_time
        elif target == 'aux1':
            return self.aux1_time
        elif target == 'aux2':
            return self.aux2_time
    """
    Return pointer to the correct buffer to edit
    """
    def _get_target_freq(self, target):
        if target == 'main':
            return self.main_freq
        elif target == 'aux1':
            return self.aux1_freq
        elif target == 'aux2':
            return self.aux2_freq
    """
    Return pointer to the correct buffer to edit
    """
    def _get_target_update(self, target):
        if target == 'main':
            return self.main_update
        elif target == 'aux1':
            return self.aux1_update
        elif target == 'aux2':
            return self.aux2_update
        elif target == 'all':
            return {'main':self.main_update,'aux1':self.aux1_update,'aux2':self.aux2_update}

    """
    Return pointer to the currect env curve to edit
    """
    def _get_target_env(self, target):
        if target == 1:
            return self.env1

    """
    Return pointer to the currect env curve point list to edit
    """
    def _get_target_env_points(self, target):
        if target == 1:
            return self.env1_points
   
    """
    Takes in a list of tuples where each tuple is in form (magnitude, angle)
    Angle is in terms of pi so 1 = pi and 1.5 = 1.5pi
    """
    @staticmethod
    def expandFactors(roots):
        #Check if empty
        if not roots:
            return [1]
        #Create factors from mag and angle of zeros/poles
        factors = [0]*len(roots)
        for i in range(len(roots)):
            factors[i] = [1, rect(abs(roots[i][0]),(roots[i][1]-1)*pi)]
            if not(roots[i][1]%2 == 0 or roots[i][1]%2 == 1):
                factors.append([1, rect(abs(roots[i][0]),-(roots[i][1]-1)*pi)])
        totalexp = [0]*(1+len(factors))
        if len(totalexp) <= 1:
            return factors
        totalexp[0] = factors[0][0]
        totalexp[1] = factors[0][1]
        length = 2
        for i in range(1,len(factors)):
            temp = totalexp[:length]
            for x in range(length):
                totalexp[x] = temp[x]*factors[i][0]
            for x in range(length):
                totalexp[x+1] += temp[x] * factors[i][1]
            length += 1

        return list(x.real for x in totalexp)

    """
    Checks if minframe and maxframe are valid and returns the correct value of minframe and maxframe if wrap-around is used

    minframe starts at 1, so must be shifted by -1 if positive.
    """
    @staticmethod
    def frameRangeCheck(minframe,maxframe,numframes):
        if maxframe <= 0:
            maxframe += numframes
            
        if minframe > 0:
            minframe -= 1
        else:
            minframe += numframes
        
        if minframe < 0 or minframe > numframes-1:
            raise RuntimeError('minframe out of bounds: %s'%minframe)
        if maxframe < 1 or maxframe > numframes+1 or maxframe <= minframe:
            raise RuntimeError('invalid maxframe bounds: %s'%maxframe)

        return minframe, maxframe
