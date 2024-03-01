#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 19:03:11 2024

@author: fangfang
"""

#Python notes
#%% import packages
import matplotlib.pyplot as plt
import math #sin, cos, tan, exp, log, floor...
import random #randint, choice, sample, shuffle
import numpy as np #np.sin, np.cos, np.tan, np.random.randint, np.random.randn ...
import pandas as pd

print(dir(math))

print(dir(random))

print(dir(np))

#%%variable types: 
"""
List
String
Set
Dictionary
Tuple
Boolean
Integer
Float
Numpy.array

"""

#List:
L0 = [1,2,3]; 
L1 = [1, 'abc', [2,3,4]]; print(L1); #lists can contain all kinds of things, even other lists
L2 = [7,8] + [3,4,5];     print(L2); #[7,8,3,4,5]
L3 = [7,8]*3; 	          print(L3); #[7,8,7,8,7,8]
L4 = [0]*5; 	          print(L4); #[0,0,0,0,0]

#list methods
L4.append(1);       print(L4)     #[0, 0, 0, 0, 0, 1]
L2.sort();          print(L2)     #[3, 4, 5, 7, 8]
num0 = L4.count(0); print(num0)   #5
L3.reverse();       print(L3)     #[8, 7, 8, 7, 8, 7]
L5 = list(range(7)); L5.remove(5); print(L5)  #[0, 1, 2, 3, 4, 6]
L5.pop(0);                         print(L5)  #[1, 2, 3, 4, 6]    
del L5[:2];                        print(L5)  #[3, 4, 6]
L5.insert(0,-1);                   print(L5)  #[-1, 3, 4, 6]

#list comprehension
L6 = [0 for i in range(10)];       print(L6)  #[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
L7 = [i*2 for i in range(10)];     print(L7)  #[0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
L8 = [[i,j] for i in range(2) for j in range(3)]; print(L8)
    #[[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]
print(len(L8))  #6
print(L8[0][1]) #0

L9 = list(range(9,2,-1));  print(L9);  #[9, 8, 7, 6, 5, 4, 3]
L10 = list(range(2,15,3)); print(L10); #[2, 5, 8, 11, 14]

#%%String:
s   = 'Hello'
ss  = "Hello"
sss = """Hello,
	nice to meet you! """
print('AB' + 'cd')      #'ABcd'
print('Hi'*4)           #'HiHiHiHi'
s = s[:3] + 'X' + s[4:] #python strings are immutable; we can't modify any part of them
print(s)                #HelXo

s1 = ['A','B','C','D','E']
s1_cat = ' '.join(s1); print(s1_cat)     #A B C D E
s2_cat = '**'.join(s1); print(s2_cat)    #A**B**C**D**E

#string methods: 
s1_cat = s1_cat.lower(); print(s1_cat);  #a b c d e
s1_cat = s1_cat.upper(); print(s1_cat);  #A B C D E
s1_cat.count('A');                       #1


#Set:
S1 = {1,2,3}
S2 = {3,4,5}

    
#%% Dictionaries
days = {'January':31, 'February':28, 'March':31, 'April':30, 'May':31, \
        'June':30, 'July':31, 'August':31, 'September':30, 'October':31,\
            'November':30, 'December':31}
print(days.keys())
days['Month0'] = 0
print(days)

#Tuple (immutable type of lists):
array1 = np.random.randn(3,5)
tuple1 = array1.shape
print(tuple1) #(3,5)
tuple2 = (1,2,3)
print(tuple) #(1,2,3)
    
    
#Booleans
bool1 = (1+2 == 3); print(bool1);     #Ttue
bool2 = (1+2 == 3.1); print(bool2) #False

#Interger
I1 = 2
I2 = int('2'); print(I2)

#Float
F1 = 2.14; 
F2 = float('2.14'); print(F2)

#%%Numpy.array
arr1 = np.array([[1,0,0],\
                 [0,1,1]])
print(arr1)
print(arr1.shape) #(2,3)
print(arr1.size)  #6

arr2 = np.linspace(1,10,10) 
print(arr2) #[ 1.  2.  3.  4.  5.  6.  7.  8.  9. 10.]

arr3 = np.arange(1,2,0.2)
print(arr3) #[1.  1.2 1.4 1.6 1.8]

arr4 = np.reshape(arr1,(3,2))
print(arr4)
#[[1 0]
# [0 0]
# [1 1]]

arr4 = np.ones((2,3,4)) #np.zeros, np.empty


#%% Syntax:
#shortcuts
"""
count = count + 1 
count += 1

total = total - 5 
total -= 5

prod = prod*2
prod *= 2

L = [1,2,3]
x,y,z = L

a, b, c = 1, 2, 3

x, y, z = y, z, x
"""

#indexing
    #0123456789
s = 'abcdefghij'
s[0]     #a
s[1]     #b
s[-1]    #j
s[-2]    #i
s[::-1]  #jihgfedcba
s[2:5]   #cde
s[:5]    #abcde
s[5:]    #fghij
s[-2:]   #ij
s[1:7:2] #bdf


#Loops
for i in range(10): 
    print(i)

l1 = 'YourName';
for items in l1:
    if items in 'aeiou':  print(items)

a = 0;
while a < 0.5:
    a = np.random.randn(1);
print(str(a) + 'is bigger than 0.5')
   
 
#%%Functions
def print_hello(n, capitalize = True, **kwargs):
    params = {
        'name': '',
        'extraThings': ''
        }
    
    #update default options with any keyword arguments provided
    params.update(kwargs)
    for i in range(n):
        s = params['name'] + '! Hello! ' + params['extraThings']
        if capitalize: s = s.upper(); 
        print(s)

print_hello(1)

print_hello(2, False, name = 'FF', extraThings = 'Goodbye!')

print_hello(1, True, name = 'FF', extraThings = 'Goodbye!')

#%%if else statement
a = np.random.randn(1);
if a[0] >= 0.9: print('>=0.9')
elif a[0] >= 0.3 and a[0] < 0.9: print('in between') 
else: print('<0.3') 

s = 'a'
if s in 'aeiou': print('Vowel') #if s == 'a' or s == 'e' or s == 'i' or s == 'o' or s == 'u'
if s not in 'aeiou': print('Not vowel')

#%%conditional operators
"""

==  #equal
>   #larger than
<   #smaller than
>=  #larger equal 
<=  #smaller equal
!=  #not equal
and 
or 
not

"""

#%%Math operators
"""
+  #addition
-  #subtraction
*  #multiplication
/  #division
** #exponentiation
// #integer division
%  #remainder

"""



