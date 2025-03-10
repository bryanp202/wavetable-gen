"""
    Depreciated!
    Do not use!
"""

import time
from cmath import rect, pi
"""
Expands out a set of factors
"""
def expandFactors(factors):
    if len(factors) == 1:
        return list(x.real for x in factors[0])

    length = len(factors)
    expansion = [0]*(length+1)

    for i in range(2**(len(factors))):
        expInd = 0
        product = 1
        for row, digit in enumerate((int(x) for x in format(i, f'0{length}b'))):
            if digit:
                expInd += 1
            else:
                product *= factors[row]
        expansion[expInd] += product    
    
    return list(x.real for x in expansion)

"""
Takes in a list of tuples where each tuple is in form (magnitude, angle)
Angle is in terms of pi so 1 = pi and 1.5 = 1.5pi
"""
def expandFactors2(roots):
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
    if len(totalexp) == 1:
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

    """
    normalize = 1/totalexp[0]
    for i in range(len(totalexp)):
        totalexp[i] *= normalize
        """

    return list(x.real for x in totalexp)
            
        
"""     
Returns string equation as numpy array with coefficients to be defactored/expanded
Do once for zeros and once for poles
"""
def getFactors(func):
    zeroOrPoles = func[1:-1]
    zeroOrPoles = zeroOrPoles.split(')*(')
    for x in range(len(zeroOrPoles)):
        data = zeroOrPoles[x].split('+z')
        zeroOrPoles[x] = complex(data[0])
    return zeroOrPoles

zeros = [[1,0.1],[1,0.2],[1,0.3]]
poles = [[0.5,0],[0.5,1]]

t = time.time()
expansion = expandFactors2(zeros)
polesexpansion = expandFactors2(poles)
dt = time.time()-t
print(dt, 'sec')
print('Zero coefficients:\n',expansion)
print('Pole coefficients:\n',polesexpansion)


wolf = f'|({expansion[0]:.6f}'
for x,zero in enumerate(expansion[1:],1):
    if zero > 0:
        wolf += '+'
    wolf += f'{zero:.6f}*e^(-i*{x}*x)'
wolf += f')/({polesexpansion[0]:.6f}'
for x,pole in enumerate(polesexpansion[1:],1):
    if pole >= 0:
        wolf += '+'
    wolf += f'{pole:.6f}*e^(-i*{x}*x)'
wolf += ')| from 0 to pi'
print(wolf)


        
