# generate random integer values
from random import seed
from random import randint
import numpy as np
# seed random number generator
seed(1)
# generate some integers
array = []
for _ in range(8000):
    a=randint(0, 100)
    b=randint(0, 100)
    c=randint(0, 100)
    d=randint(0, 100)
    e=randint(0, 100)
    print(e)
    g = a * b + c * d + e 
    array.append([a, b, c, d, e, g])
    
a = np.asarray(array, dtype=np.uint8)
np.savetxt("testdatalarge.csv", a.astype(int),fmt='%i', delimiter='\t')   
