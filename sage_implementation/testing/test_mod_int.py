#!/usr/bin/env python2

import sys
import math
import time

print 'Compute x mod n in Finite Field:'
print 'y such that xy = 1 mod n'
x_input = int(raw_input('Input X: '))
n_input = int(raw_input('Input N: '))

def test_function(x = 0, n = 0):
    tmp = 0
    a = 0
    b = 0
    last_t = 0
    t = 0
    next_t = 0
    q = 0
    counter = 0

    if n == 1:
        return 0
    a = n
    b = x
    t = 0
    next_t = 1
    while b != 0:
        counter += 1
        # a = s * n + t * x
        if b == 1:
            next_t = next_t % n
            if next_t < 0:
                next_t = next_t + n
            return next_t
        q = a / b
        tmp = b
        b = a % b
        a = tmp
        last_t = t
        t = next_t
        next_t = last_t - q * t
    raise ZeroDivisionError("inverse of Mod({x}, {n}) does not exist")

start = time.clock()
print (test_function(x_input, n_input))
end = time.clock()
print "Time: %f"%(float(end-start))