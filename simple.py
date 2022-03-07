#!/usr/bin/env python3

from random import randint

def gen_A(*args):
    return randint(0, 2**32 - 1)

def gen_B(*args):
    A = args[0]
    return (A&0xffff0000) + randint(0, 2**16 - 1)

def gen_C(*args):
    B = args[0]
    return (B&0x0000ffff) +(randint(0, 2**16 - 1) << 16)

A = []
B = []
C = []

for i in range(100):
    A.append(gen_A())
    B.append(gen_B(A[-1]))
    C.append(gen_C(B[-1]))

print(A)
print(B)
print(C)