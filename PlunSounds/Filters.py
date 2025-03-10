from numpy import array
import time
def expandFactors(factors):
    if len(factors) == 1:
        return list(x.real for x in factors[0])
    
    length = len(factors)
    totalexpansion = [1]*(2**(len(factors)))
    rn = len(factors[0])

    for out in range(length):
        for time in range(int(rn ** (out))):
            for num in range(rn):
                for l in range(rn ** (length - 1) // (rn ** out)):
                    totalexpansion[l + num * rn ** (length - 1)//(rn**out) + time * rn ** (length-out)] *= factors[out][num]

    expansion = [0]*(len(factors)+1)
    for i in range(len(totalexpansion)):
        expansion[sum(int(item) for item in "{0:b}".format(i))] += totalexpansion[i]
    
    
    return list(x.real for x in expansion)
"""
Returns string equation as numpy array with coefficients to be defactored/expanded
Do once for zeros and once for poles
"""
def getFactors(func):
    zeroOrPoles = func[1:-1]
    zeroOrPoles = zeroOrPoles.split(')*(')
    for x in range(len(zeroOrPoles)):
        data = zeroOrPoles[x].split('+z')
        zeroOrPoles[x] = (complex(data[0]),1.0+0.0j)
    return zeroOrPoles

t = time.time()
zeros = getFactors('(-0.5+0.5j+z)*(-0.5-0.5j+z)*(-0.5+0.3j+z)*(-0.5-0.3j+z)*(-0.5+0.5j+z)*(-0.5-0.5j+z)*(-0.5+0.3j+z)*(-0.5-0.3j+z)')
t = time.time()
expansion = expandFactors(zeros)
dt = time.time()-t
print(dt)
print(len(zeros))
print('expansion:\n',expansion)
        
