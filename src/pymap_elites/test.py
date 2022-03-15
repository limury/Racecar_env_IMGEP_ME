import numpy as np

def f1():
    print('hi')
def f2():
    print('bye')
def f3():
    print('hola')


a = { f1: 0.1, f2: 0.1, f3: 0.8}

x = np.random.choice(a.keys(), size=10, p=[ a[f] for f in a ])

print(x)