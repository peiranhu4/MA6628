# MA6628

# Prj01 (L01)

import math
import sys
def isprime(n):
    for x in range(3,int(n**0.5)+1,2):
        if n%x == 0:
            return 0
    return 1
if __name__ == "__main__":
    a,b=1000,1002
    count =0
    for x in range(1000,1000000):
        if isprime(x)*isprime(x+2):
            a,b=x,x+2
            count = count +1
    print('The total number of twin primes between 1000 and 1000000 is %d' %count)
    print('The biggest twin prime I could find is %d %d' %(a,b))
The total number of twin primes between 1000 and 1000000 is 18462
The biggest twin prime I could find is 999959 999961
