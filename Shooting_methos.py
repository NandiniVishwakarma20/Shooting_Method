import numpy as np
import matplotlib.pyplot as plt

def f(x, y, v):
    return 2 * y + 8 * x * (9 - x)

def runge_kutta(h, x_range, y0, v0):
    n = len(x_range)
    y = np.zeros(n)
    v = np.zeros(n)
    y[0], v[0] = y0, v0
    
    for i in range(n - 1):
        x = x_range[i]
        k1y = h * v[i]
        k1v = h * f(x, y[i], v[i])
        
        k2y = h * (v[i] + 0.5 * k1v)
        k2v = h * f(x + 0.5 * h, y[i] + 0.5 * k1y, v[i] + 0.5 * k1v)
        
        k3y = h * (v[i] + 0.5 * k2v)
        k3v = h * f(x + 0.5 * h, y[i] + 0.5 * k2y, v[i] + 0.5 * k2v)
        
        k4y = h * (v[i] + k3v)
        k4v = h * f(x + h, y[i] + k3y, v[i] + k3v)
        
        y[i + 1] = y[i] + (k1y + 2 * k2y + 2 * k3y + k4y) / 6
        v[i + 1] = v[i] + (k1v + 2 * k2v + 2 * k3v + k4v) / 6
    
    return y[-1] 

def shooting_method(a, b, al, be, s1, s2, tol=1e-6, max_iter=100):
    x_range = np.linspace(a, b, 100)
    h = x_range[1] - x_range[0]
    
    for _ in range(max_iter):
        y1 = runge_kutta(h, x_range, al, s1)
        y2 = runge_kutta(h, x_range, al, s2)
        
        s_new = s2 - (y2 - be) * (s2 - s1) / (y2 - y1) 
        
        if abs(y2 - be) < tol:
            return s2, x_range, runge_kutta_values(x_range, al, s2)
        
        s1, s2 = s2, s_new
    
    raise ValueError("Shooting method did not converge")

def runge_kutta_values(x_range, y0, v0):
    h = x_range[1] - x_range[0]
    n = len(x_range)
    y = np.zeros(n)
    v = np.zeros(n)
    y[0], v[0] = y0, v0
    
    for i in range(n - 1):
        x = x_range[i]
        k1y = h * v[i]
        k1v = h * f(x, y[i], v[i])
        
        k2y = h * (v[i] + 0.5 * k1v)
        k2v = h * f(x + 0.5 * h, y[i] + 0.5 * k1y, v[i] + 0.5 * k1v)
        
        k3y = h * (v[i] + 0.5 * k2v)
        k3v = h * f(x + 0.5 * h, y[i] + 0.5 * k2y, v[i] + 0.5 * k2v)
        
        k4y = h * (v[i] + k3v)
        k4v = h * f(x + h, y[i] + k3y, v[i] + k3v)
        
        y[i + 1] = y[i] + (k1y + 2 * k2y + 2 * k3y + k4y) / 6
        v[i + 1] = v[i] + (k1v + 2 * k2v + 2 * k3v + k4v) / 6
    
    return y

a = 0
b = np.pi
al =  0
be = 1
s1 = 0
s2 = 2 

s_final, x_values, y_values = shooting_method(a, b, al, be, s1, s2) 

plt.plot(x_values, y_values, label="Numerical Solution")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Shooting Method Solution")
plt.legend()
plt.show()

print(s_final)
