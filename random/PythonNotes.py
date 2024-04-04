#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 19:03:11 2024

@author: fangfang
"""

#Python notes
#%% import packages
#packages are collections of modules that include functions, methods, and types
#that allow you to perform many tasks without having to write your code from scratch.
import matplotlib.pyplot as plt
import math #sin, cos, tan, exp, log, floor...
import random #randint, choice, sample, shuffle

# The 'numpy' package provides support for arrays, matrices, and a large collection of
# high-level mathematical functions to operate on these arrays.
import numpy as np #np.sin, np.cos, np.tan, np.random.randint, np.random.randn ...

print(dir(math))

print(dir(random))

print(dir(np))

#we can also import functions from other scripts

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

#List: A list is a collection which is ordered and changeable. Allows duplicate members.
#Creating simple lists
L0 = [1,2,3]; 
L1 = [1, 'abc', [2,3,4]]; print(L1); #lists can contain all kinds of things, even other lists
L2 = [7,8] + [3,4,5];     print(L2); #[7,8,3,4,5]
L3 = [7,8]*3; 	          print(L3); #[7,8,7,8,7,8]
L4 = [0]*5; 	          print(L4); #[0,0,0,0,0]

#list methods to modify lists
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

# Creating lists with specific ranges
# Creates a list of numbers from 9 down to 3, decreasing by 1 
L9 = list(range(9,2,-1));  print(L9);  #[9, 8, 7, 6, 5, 4, 3] (it doesn't include 2)
L10 = list(range(2,15,3)); print(L10); #[2, 5, 8, 11, 14]

#%%String:
# Strings in Python can be defined using single quotes, double quotes, or 
#triple quotes for multi-line strings.
s   = 'Hello'
ss  = "Hello"
sss = """Hello,
	nice to meet you! """
print('AB' + 'cd')      #'ABcd'
print('Hi'*4)           #'HiHiHiHi'
s = s[:3] + 'X' + s[4:] #python strings are immutable; we can't modify any part of them
print(s)                #HelXo

s1 = ['A','B','C','D','E']
s1_cat = ' '.join(s1);  print(s1_cat)     #A B C D E
s2_cat = '**'.join(s1); print(s2_cat)    #A**B**C**D**E

#string methods: 
s1_cat = s1_cat.lower(); print(s1_cat);  #a b c d e
s1_cat = s1_cat.upper(); print(s1_cat);  #A B C D E
s1_cat.count('A');                       #1
    
#%% Dictionaries
# Defining a dictionary with month names as keys and the number of days as values.
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

#Set: 
#Sets are unordered collections. The order of items is undefined, so you 
#cannot index or access elements by position. Sets cannot contain duplicate 
#elements. Adding a duplicate item to a set will not raise an error, but the 
#set will not change.
S1 = {1,2,3}
S2 = {3,4,5}
#can't do S1[0]
S1.add(5);   print(S1); #{1, 2, 3, 5}
S1.add(1);   print(S1); #{1, 2, 3, 5} 
    
#Booleans
bool1 = (1+2 == 3); print(bool1);     #True
bool2 = (1+2 == 3.1); print(bool2) #False

#Interger
I1 = 2
I2 = int('2'); print(I2)

#Float
F1 = 2.14; 
F2 = float('2.14'); print(F2)

#%%Numpy.array
# Creating a 2D numpy array
arr1 = np.array([[1,0,0],\
                 [0,1,1]])
print(arr1)
print(arr1.shape) #(2,3)
print(arr1.size)  #6

# Creates an array of 10 numbers evenly spaced between 1 and 10
arr2 = np.linspace(0,30,6) 
print(arr2) #[ 0.  6. 12. 18. 24. 30.]

# Creating an array with numbers starting from 1 to less than 2 with a step of 0.2
arr3 = np.arange(1,2,0.2)
print(arr3) #[1.  1.2 1.4 1.6 1.8]

# Reshaping an existing array to a new shape
arr4 = np.reshape(arr1,(3,2))
print(arr4)
#[[1 0]
# [0 0]
# [1 1]]

# Creating a 3D array of ones
arr4 = np.ones((2,3,4)) #np.zeros, np.empty
#array([[[1., 1., 1., 1.],
#        [1., 1., 1., 1.],
#        [1., 1., 1., 1.]],
#
#       [[1., 1., 1., 1.],
#        [1., 1., 1., 1.],
#        [1., 1., 1., 1.]]])

# Creating a diagonal matrix
diag_arr = np.diag([1, 2, 3])
print(diag_arr)
# Prints a diagonal matrix:
# [[1 0 0]
#  [0 2 0]
#  [0 0 3]]

# Creating an identity matrix
identity_matrix = np.eye(3)
print(identity_matrix)
# Prints a 3x3 identity matrix:
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]

# Random array generation
random_arr = np.random.rand(3, 2)  # Creates a 3x2 array of random floats in the half-open interval [0.0, 1.0)
print(random_arr)


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
    
    """
    Parameters:
        n: The number of times a greeting should be printed.
        capitalize: A boolean parameter (defaulted to True) that determines 
            if the output string should be converted to uppercase.
        **kwargs: A catch-all for any number of additional named arguments not 
            explicitly defined in the function signature. These arguments are 
            captured as a dictionary.
    """
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
    
    return s, params

#print_hello(1)

#_, param1 = print_hello(2, False, name = 'FF', extraThings = 'Goodbye!')

print_hello(1, True, name = 'FF', extraThings = 'Goodbye!')

#%%if else statement
a = np.random.randn(1);
if a[0] >= 0.9: print('>=0.9')
elif a[0] >= 0.3 and a[0] < 0.9: print('in between')  # 0.3<= a[0] < 0.9
else: print('<0.3') 

s = 'a'
if s in 'aeiou': print('Vowel') #if s == 'a' or s == 'e' or s == 'i' or s == 'o' or s == 'u'
if s not in 'aeiou': print('Not vowel')

#%% class


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

#%% FOR NEXT TIME
"""
Matplotlib
NumPy
Pandas
Scikit-learn
PyTorch
TensorFlow
Keras
SciPy
Seaborn

"""

#%% 
"""JAX  accelerated NumPy
Fundamentally, JAX is a library that enables transformations of array
-manipulating programs written with a NumPy-like API. We can think of JAX as 
differentiable NumPy that runs on accelerators.
"""
import numpy as np
import jax
import jax.numpy as jnp

x = jnp.arange(10)
print(x)
#Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int32)

def sum_of_squares(x):
    return jnp.sum(x**2)

"""
Applying jax.grad to sum_of_squares will return a different function, namely 
the gradient of sum_of_squares with respect to its first parameter x.
Then, we can use that function on an array to return the derivatives with 
respect to each element of the array.
"""
sum_of_squares_dx = jax.grad(sum_of_squares)
x = jnp.asarray([1.0, 2.0, 3.0, 4.0])
print(sum_of_squares(x))  #30
print(sum_of_squares_dx(x))  #[2. 4. 6. 8.]

#By default, jax.grad will find the gradient with respect to the first 
#argument. In the example below, the result of sum_squared_error_dx will be 
#the gradient of sum_squared_error with respect to x.
def sum_squared_error(x,y):
    return jnp.sum((x-y)**2)

sum_squared_error_dx = jax.grad(sum_squared_error)
y = jnp.asarray([1.1, 2.1, 3.1, 4.1])
print(sum_squared_error_dx(x,y)) #[-0.2, -0.2, -0.2, -0.2]

#To find the gradient with respect to a different argument (or several), 
#you can set argnums:
sum_squared_error_dy = jax.grad(sum_squared_error, argnums = (1))
print(sum_squared_error_dy(x,y))
sum_squared_error_dx_dy = jax.grad(sum_squared_error, argnums=(0, 1))
print(sum_squared_error_dx_dy(x,y))
#(Array([-0.2, -0.2, -0.2, -0.2], dtype=float32),
# Array([0.2, 0.2, 0.2, 0.2], dtype=float32))


"""
value and grad: Often, you need to find both the value and the gradient of a 
#function, e.g. if you want to log the training loss. JAX has a handy sister 
#transformation for efficiently doing that
"""
sum_squared_error_val_dx_dy = jax.value_and_grad(sum_squared_error, argnums = (0,1))
print(sum_squared_error_val_dx_dy(x,y))
#(Array(0.04, dtype=float32),
#Array([-0.2, -0.2, -0.2, -0.2], dtype=float32))


"""Auxiliary data: In addition to wanting to log the value, we often want to 
#report some intermediate results obtained in computing the loss function.
"""
def squared_error_with_aux(x,y):
    return sum_squared_error(x,y), x-y
#has_aux signifies that the function returns a pair, (out, aux). It makes 
#jax.grad ignore aux, passing it through to the user, while differentiating the 
#function as if only out was returned.
sum_squared_error_dx_aux = jax.grad(squared_error_with_aux, has_aux = True)
print(sum_squared_error_dx_aux(x,y))
#(Array([-0.2, -0.2, -0.2, -0.2], dtype=float32),
#Array([-0.1, -0.1, -0.1, -0.1], dtype=float32))

#Differences from numpy
x = np.array([1,2,3])
x[0] = 123
print(x) #[123, 2, 3]

y = jnp.array([1,2,3])
#y[0] = 123 this will throw an error
y = y.at[0].set(123) #if we just do y.at[0].set(123) and print out y, we still get [1,2,3]
print(y) #[123, 2,3]

""" using grad() to fit a model """
xs = np.random.normal(size=(100,))
noise = np.random.normal(scale = 1, size=(100,))
w_true, b_true = 3, -1
ys = xs*w_true + b_true + noise
plt.scatter(xs, ys)

def model(theta, x):
    w, b = theta
    return w*x + b

def loss_func(theta, x, y):
    prediction = model(theta, x)
    return jnp.mean((prediction - y)**2)

#How do we optimize a loss function? Using gradient descent. At each update 
#step, we will find the gradient of the loss w.r.t. the parameters, and take a 
#small step in the direction of steepest descent:
#theta_{new} = theta - 0.1* derivative with respect to theta
def update(theta, x, y, lr = 0.1):
    return theta - lr * jax.grad(loss_func)(theta, x, y)

theta = jnp.array([1.0,1.0])
for _ in range(1000):
    theta = update(theta, xs, ys)

plt.scatter(xs,ys)
plt.plot(xs, model(theta,xs))
w_est, b_est = theta
print(f"w: {w_est:<.2f}, b: {b_est: <.2f}")

#%% 
""" Automatic vectorization in JAX
"""
x = jnp.arange(5) #Array([0, 1, 2, 3, 4], dtype=int32)
w = jnp.array([2.0, 3.0, 4.0])

def convolve(x,w):
    output = []
    for i in range(1, len(x)-1):
        output.append(jnp.dot(x[i-1:i+2], w))
    return jnp.array(output)

convolve(x,w) #Array([11., 20., 29.], dtype=float32)

#suppose we would like to apply this function to a batch of weights w to a 
#batch of vectors x
x2 = jnp.arange(3,8)
xs = jnp.stack([x,x2])
#Array([[0, 1, 2, 3, 4],
#       [3, 4, 5, 6, 7]], dtype=int32)
w2 = jnp.array([5.0, 6.0, 7.0])
ws = jnp.stack([w,w2])
#Array([[2., 3., 4.],
#       [5., 6., 7.]], dtype=float32)

#we could apply a loop over the batch
def manually_batched_convolve(xs, ws):
    output = []
    for i in range(xs.shape[0]):
        output.append(convolve(xs[i], ws[i]))
    return jnp.stack(output)
manually_batched_convolve(xs, ws)
#Array([[ 11.,  20.,  29.],
#       [ 74.,  92., 110.]], dtype=float32)

"""
In JAX, the jax.vmap transformation is designed to generate such a vectorized 
implementation of a function automatically.

By default, jax.vmap assumes in_axes=0 for all inputs if not specified. This 
means the function convolve is mapped over the first dimension (axis 0) of both 
xs and ws, treating this dimension as the batch dimension! The output is 
straightforward: it performs the convolution for each pair of x and w in the 
batches, maintaining the order and structure of the inputs.

"""
auto_batch_convolve = jax.vmap(convolve)
auto_batch_convolve(xs, ws)
#Array([[ 11.,  20.,  29.],
#       [ 74.,  92., 110.]], dtype=float32)

"""
The in_axes argument specifies which axes of the input tensors should be 
considered as the batch dimensions to be mapped over. It dictates how the 
inputs to the function being vectorized should be parallelized.

If we set in_axes=1, it means that the function will be mapped over the 2nd 
dimension of the input arrays. 

If we set out_axes=1, the result is then structured such that the batch 
dimension is along axis 1 in the output, matching the specified out_axes.

"""
auto_batch_convolve_v2 = jax.vmap(convolve, in_axes = 1, out_axes = 1)
xst = jnp.transpose(xs)
wst = jnp.transpose(ws)
auto_batch_convolve_v2(xst, wst)
#Array([[ 11.,  74.],
#       [ 20.,  92.],
#       [ 29., 110.]], dtype=float32)
#if we set out_axes = 0
#Array([[ 11.,  20.,  29.],
#       [ 74.,  92., 110.]], dtype=float32)

""" 
This configuration tells jax.vmap to vectorize over the first dimension of 
the first argument (xs), treating it as the batch dimension, while treating 
the second argument (w) as a non-batched, or broadcasted, argument. This means 
the same w is used for each convolution operation across the batch of xs, 
rather than using a corresponding pair from ws. 

"""
auto_batch_convolve_v3 = jax.vmap(convolve, in_axes = [0,None])
auto_batch_convolve_v3(xs, w)
#Array([[11., 20., 29.],
#       [38., 47., 56.]], dtype=float32)

#%% Just in time compilation with JAX
def selu(x, alpha = 1.67, lambda_ = 1.05): #Scaled Exponential Linear Unit
    # For elements of x that are positive, it simply returns the element itself 
    #(scaled by lambda_). For non-positive elements, it applies an exponential 
    #function scaled by alpha and lambda_, then subtracts alpha to shift the 
    #curve downward.
    print(x)
    #Traced<ShapedArray(int32[10])>with<DynamicJaxprTrace(level=1/0)>
    return lambda_ * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)
    #Compatible with JIT because it's a vectorized operation designed for array 
    #inputs, allowing JAX to analyze and optimize the operation statically.
x = jnp.arange(10)

"""
jax.jit(selu) compiles the selu function using JAX's JIT compilation, resulting 
in selu_jit. This compilation process translates the Python function into 
highly efficient machine code tailored for the specific operation and the 
hardware it's running on (e.g., CPU, GPU, or TPU).

Here’s what just happened:

We defined selu_jit as the compiled version of selu.

We called selu_jit once on x. This is where JAX does its tracing – it needs to 
have some inputs to wrap in tracers, after all. The jaxpr is then compiled 
using XLA into very efficient code optimized for your GPU or TPU. Finally, the 
compiled code is executed to satisfy the call. Subsequent calls to selu_jit 
will use the compiled code directly, skipping the python implementation 
entirely.

"""
selu_jit = jax.jit(selu)
selu_jit(x)
#Array([0.       , 1.05     , 2.1      , 3.1499999, 4.2      , 5.25     ,
#       6.2999997, 7.3499994, 8.4      , 9.45     ], dtype=float32)


"""
When you apply jax.jit to a function without specifying any arguments as 
static, JAX attempts to trace the function to create a compiled version that 
can be efficiently executed on accelerators like GPUs or TPUs. During this 
tracing process, JAX replaces the inputs with abstract placeholders. This 
allows JAX to understand the shape and data type of inputs BUT NOT THEIR
ACTUAL VALUES.

The if x > 0 condition in the function f(x) requires knowing the actual value 
of x to decide which branch of the code to execute. However, because JAX is 
tracing the function with abstract placeholders, it cannot determine the value 
of x at compile time. This inability to resolve the condition based on value 
leads to an error, as JAX cannot compile a function where the control flow 
depends on a value that is unknown at compile time.
"""
# Condition on value of x.

def f(x):
  if x > 0:
    return x
  else:
    return 2 * x

f_jit = jax.jit(f)
#f_jit(10)  # Should raise an error. 

def g(x, n):
  i = 0
  while i < n:
    i += 1
  return x + i

g_jit = jax.jit(g)
#g_jit(10, 20)  # Should raise an error. 

"""
The loop_body function is decorated with @jax.jit, which means this function 
is JIT-compiled by JAX. JIT (Just-In-Time) compilation is a process where the 
function's Python code is compiled into highly efficient machine code the first 
time it's called. Subsequent calls to this function are much faster because 
they can bypass the Python interpreter and execute the compiled machine code 
directly.

The while loop inside the g_inner_jitted function, including its control flow 
(i < n) and the call to loop_body(i), is not JIT-compiled. This is because the 
loop itself is controlled by Python's native control flow constructs, which 
are executed by the Python interpreter.

"""
@jax.jit
def loop_body(prev_i):
  return prev_i + 1

def g_inner_jitted(x, n):
  i = 0
  while i < n:
    i = loop_body(i)
  return x + i

g_inner_jitted(10, 20)

#When g_inner_jitted(10, 20) is called, the function enters a Python-controlled 
#while loop, calling the JIT-compiled loop_body at each iteration. While the 
#loop condition and increment are handled by Python, the increment operation 
#benefits from JIT compilation's speed due to loop_body being compiled.

"""
If we really need to JIT a function that has a condition on the value of an 
input, we can tell JAX to help itself to a less abstract tracer for a 
particular input by specifying static_argnums or static_argnames. The cost of 
this is that the resulting jaxpr is less flexible, so JAX will have to 
re-compile the function for every new value of the specified static input. 
It is only a good strategy if the function is guaranteed to get limited 
different values.

"""
f_jit_correct = jax.jit(f, static_argnums=0)
print(f_jit_correct(10))

#Static Handling of n: It tells JAX that the function may be compiled 
#differently for different values of n, but within each compiled version, the 
#value of n is fixed and known. This allows JAX to handle the loop correctly 
#by "unrolling" it for each specific value of n at compile time, effectively 
#creating a version of the function where the loop's behavior is statically 
#determined.
g_jit_correct = jax.jit(g, static_argnames=['n'])
print(g_jit_correct(10, 20))





