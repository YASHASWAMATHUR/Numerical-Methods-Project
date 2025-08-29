import numpy 

def dose(w, D_measured, D_prescribed):
    return w * D_measured - D_prescribed

def dose_prime(w, D_measured): 
    return D_measured

D_measured = 1.5
D_prescribed = 10
w0 = 5.0
tol = 1e-6

for i in range(10):
    f = dose(w0, D_measured, D_prescribed)
    f_prime = dose_prime(w0, D_measured)
    w1 = w0 - f / f_prime
    print(f"Iteration {i+1}: w = {w1}, f = {f}")
    if abs(w1 - w0) < tol: 
        break
    w1 = w0 


