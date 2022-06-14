# Computational Mathematics Project: Integral Equation
This is a piece of coursework I did as a part of my Computational Mathematics module. It has a merge of Calculus and Python to allow for numerical analysis.

# MAT 20031 Project: Integral equation

### Student number: 18004543

## Introduction
In this project, we'll be attempting will be attempting to look for the solution of $x$ within the following equation which will result in finding the interval that gives the required value of the integral:

$$ \int_{0}^{x}{\frac{1}{\sqrt{2\pi}}e^{-t^2/2}dt}\ =\ 0.45 $$

I'm interested in this problem because I'm curious about the computational accuracy of finding integrals when using different algorithms such Newton's method for computing roots and Simpson's rule for computing definite integrals.

### Definitions and methodology
#### Newton's method
We first want to attempt to use Newton's method to have a search for a root. This is defined as:

$$x_{n+1}=x_n-\frac{f(x_n)}{f'(x_n)}$$

For this to be carried out, we need to find $f(x)$ and $f'(x)$.

#### Defining $f(x)$
Here we move everything to the left-hand side.

$$ f(x) = \int_0^x \dfrac{1}{\sqrt{2 \pi}} e^{-t^2/2}~dt - 0.45 = 0$$

$$ f(x) = \int_0^x \dfrac{1}{\sqrt{2 \pi}} e^{-t^2/2}~dt - 0.45$$

The integral within our $f(x)$ will be computed from our Simpson's rule which we will discuss later.

#### Defining $f'(x)$
We can work our $f(x)$ around to $f'(x)$:

$$ f^\prime\left(x\right)=\frac{e^{-x^2/2}}{\sqrt{2\pi}} $$

#### Simpsons Rule
This is our definition for a composite Simpson's rule:

$$ \int_a^b f(x) dx = \frac{h}{3}\sum_{i=0}^{N/2}\left[f(x_{2i} + 4x_{2i+1} + x_{2i+2} \right] + E_i  $$

### Defining the true value of $x$
To have a look at the accuracy, we need to do the integral by hand and look for $x$ to make a comparison between the calculated value and the true value. The true value will be used to compute the error between the calculated value and its true value.

$$ \int_{0}^{x}{\frac{1}{\sqrt{2\pi}}e^{-t^2/2}dt}\ =\ 0.45 $$

$$ \left[\frac{1}{2}erf\left(\frac{t}{\sqrt2}\right)\right]_0^x=0.45 $$

$$ \left[\frac{1}{2}erf\left(\frac{x}{\sqrt2}\right)\right]-\ \left[\frac{1}{2}erf\left(0\right)\right]=0.45 $$

$$ \left[\frac{1}{2}erf\left(\frac{x}{\sqrt2}\right)\right]-\ 0=0.45 $$

$$ erf\left(\frac{x}{\sqrt2}\right)=0.9 $$

$$ erf^{-1}\left(0.9\right)=\frac{x}{\sqrt2} $$

$$ \sqrt2erf^{-1}\left(0.9\right)=x $$

We can compute $ \sqrt2erf^{-1}\left(0.9\right)=x $ using the scipy module and use the inverse error function to compute $ x $. I came across 'erfinv' function within the scipy documentation while looking for a route to invert the error function within our result for the true value.

We can quickly view a preview of the true value now using a function to see what we are aiming for. This function can also act as a short hand way of accessing the true value in later error calculations to reduce clutter within the code.

Before we do so, we need to import our dependencies for our Python work.

In the following code, we will import dependencies and implement our definitions for Newton's method and the Simpson's rule.
### Importing dependencies
First, we are setting up with importing all the modules that we need.


```python
%matplotlib notebook
import numpy as np
from scipy import special #Used to access erfinv functions
```

Now we can show the preview of our true value:


```python
def trueX():
    return np.sqrt(2)*special.erfinv(0.9)

print("True value of x is:",trueX())
```

    True value of x is: 1.6448536269514724


## Implementation into Python

### Setting up test parameters and their initial values
Here I'm going to set up the inital value for x, the tolerance that is going to be used and the number of max interations that are going to be carried out for Newton's method. Also including a parameter that will be used within the Simpson's rule method to determine its sub intervals.

All of these values can and will be changed during the running of the program to test different parameters and see how they affect our computed value of $x$ compared to its true value.

Here is the declarion of our testing values and their initial values:


```python
initialX = 1 #Definition for the x0 we shall be using for our Newton's method 
tolerance =  10**(-5) #Tolerance for which xn+1 will be tested compared to 0
maxIterations = 100 #Max iterations of how many times our Newton's method
simpSubIntervals = 21 #Number of sub intervals for the Simpson's rule. This value must be odd
```

### Defining Simpson's rule
Here we define the composite Simpson's rule. Our Simpson's rule can have a function passed in for it to integrate.


```python
#Simpson's rule function with the parameters: the function to have its integral computed,integral bound A, integral bound B, sub intervals 
def simpsons(f,boundA,boundB,N):
    #Setting h
    h = (boundB-boundA)/(N-1)
    #Interpolating values between the bounds at N intervals
    x = np.linspace(boundA,boundB,N)
    
    #Simpsons rule embedded with our f(x)
    simpo = (h/3) * np.sum(f(x[0:N-2:2]) + 4*f(x[1:N-1:2]) + f(x[2:N:2]))
    #Return Simpson's rule output
    return simpo
```

### Defining $f(x)$

$$ f(x) = \int_0^x \dfrac{1}{\sqrt{2 \pi}} e^{-t^2/2}~dt - 0.45$$

Here is the definition for our $f(x)$. As you can see the Simpson's rule we defined earlier is being used here to compute our integral(defined in the arguments of the function). This is the key to the whole project as we are computing the integral using Simpson's rule.


```python
def f(x):    
    return simpsons(lambda x:np.exp(-(x**2)/2)/np.sqrt(2*np.pi),0,x,simpSubIntervals) - 0.45
```

### Defining $f'(x)$

$$ f^\prime\left(x\right)=\frac{e^{-x^2/2}}{\sqrt{2\pi}} $$

DefinIng our $f'(x)$ using are calculation within the definitions. We are taking advantage of numpy functions here.


```python
def df(x):
    return np.exp(-(x**2)/2)/np.sqrt(2*np.pi)
```

### Defining Newton's method

$$x_{n+1}=x_n-\frac{f(x_n)}{f'(x_n)}$$

This is the most important function which pulls alot of our definitions together to find our computed value. While also performing Newton's iterations it also displays how the values change with each iteration and its final output. All initial test values are pulled from our definitions, these can be changed. It also returns the final computer value of $x$ which will later be compared to the true value of $x$.


```python
#Newton's method function with the parameters: initial value, tolerance and max iterations.
def newtons():
    xn = initialX #Setting initial value of xn to initialX
    
    print("n          x_n+1          x_n          f(x_n)          f'(x_n)")
    print("____________________________________________________________________")#Formatting for printing each iteration of the loop
    
    for i in range(maxIterations):
        xnp1 = xn - (f(xn)/df(xn)) #Calculating x_n+1
        
        print("{:d} {:.10e} {:.10e} {:.10e} {:.10e}".format(i,xnp1,xn,f(xn),df(xn))) #Formatting for table
        
        if(abs(f(xnp1))<tolerance):
            print("") #Space
            print("Our calculated estimated value of x is",xnp1,"at a Newton's method tolerance of",tolerance,"at",simpSubIntervals,"Simpsons rule subintervals")
            return xnp1 #Return final value
        
        xn = xnp1
```

### Accuracy of integration and complete function
Here we'll define a fuction that can be used to quickly determine the error between our calcutlated value based on our set parameters and the true value of $x$. This is essentially one large function that compliles all of our previous functions together to nicely output all the values together with are test parameters to speed up testing different parameters.


```python
def compFunction():
    computedValue = newtons()
    print("")
    print("Difference between calculated value and true value of x is:",(abs(trueX()-computedValue)))
```

## Experimentation
### Testing values for converage to true $x$
Now we can begin testing and expermienting differant parameters and analyse the convergance. First I'll test with the default values that we set previously:

initialX = 1

tolerance =  10**(-5)

maxIterations = 100

simpSubIntervals = 21


```python
compFunction()
```

    n          x_n+1          x_n          f(x_n)          f'(x_n)
    ____________________________________________________________________
    0 1.4490429052e+00 1.0000000000e+00 -1.0865523711e-01 2.4197072452e-01
    1 1.6185177051e+00 1.4490429052e+00 -2.3662772648e-02 1.3962413682e-01
    2 1.6442971375e+00 1.6185177051e+00 -2.7755215266e-03 1.0766418286e-01
    3 1.6448532503e+00 1.6442971375e+00 -5.7407565600e-05 1.0323007216e-01
    
    Our calculated estimated value of x is 1.6448532503375997 at a Newton's method tolerance of 1e-05 at 21 Simpsons rule subintervals
    
    Difference between calculated value and true value of x is: 3.7661387275456093e-07


With a fairly low initial tolerance and high number of sub-intervals we seem to have a strong convergence to the true value of x.

Now we'll experiement by reducing the sub-intervals in the Simpson's rule and increasing tolerance for the Newton's iteration to see if the value tapers away from our true value.


```python
initialX = 1
tolerance =  10**(-2)
maxIterations = 100
simpSubIntervals = 5
```


```python
compFunction()
```

    n          x_n+1          x_n          f(x_n)          f'(x_n)
    ____________________________________________________________________
    0 1.4489985818e+00 1.0000000000e+00 -1.0864451214e-01 2.4197072452e-01
    1 1.6183867629e+00 1.4489985818e+00 -2.3652197596e-02 1.3963310454e-01
    
    Our calculated estimated value of x is 1.6183867628619013 at a Newton's method tolerance of 0.01 at 5 Simpsons rule subintervals
    
    Difference between calculated value and true value of x is: 0.026466864089571107


Here we can see it diverging from the true value. Lets again change some more values and see if we can divert more.


```python
initialX = 1
tolerance =  10**(-1)
maxIterations = 10
simpSubIntervals = 3
```


```python
compFunction()
```

    n          x_n+1          x_n          f(x_n)          f'(x_n)
    ____________________________________________________________________
    0 1.4482812878e+00 1.0000000000e+00 -1.0847094800e-01 2.4197072452e-01
    
    Our calculated estimated value of x is 1.4482812878262994 at a Newton's method tolerance of 0.1 at 3 Simpsons rule subintervals
    
    Difference between calculated value and true value of x is: 0.19657233912517302


Again, we see the value moving away from what we expect. Now let's try changing the values to get strong accuracy.


```python
initialX = 1
tolerance =  10**(-10)
maxIterations = 1000
simpSubIntervals = 101
```


```python
compFunction()
```

    n          x_n+1          x_n          f(x_n)          f'(x_n)
    ____________________________________________________________________
    0 1.4490429746e+00 1.0000000000e+00 -1.0865525390e-01 2.4197072452e-01
    1 1.6185179213e+00 1.4490429746e+00 -2.3662790766e-02 1.3962412278e-01
    2 1.6442972917e+00 1.6185179213e+00 -2.7755138771e-03 1.0766414519e-01
    3 1.6448533723e+00 1.6442972917e+00 -5.7404229776e-05 1.0323004599e-01
    4 1.6448536268e+00 1.6448533723e+00 -2.6239189654e-08 1.0313568357e-01
    
    Our calculated estimated value of x is 1.6448536267545582 at a Newton's method tolerance of 1e-10 at 101 Simpsons rule subintervals
    
    Difference between calculated value and true value of x is: 1.9691426267343104e-10


Now we have a highly accurate result within 10dp of the true value.

## Conclusion
Putting all of this together, we have managed to create computational functions of differant methods such as Newton's and Simpson's rule to reverse engineer an integral within a function to find its upper bound. We've been able to modify our test parameters to increase or decrease the accuracy of computed value of $x$.
