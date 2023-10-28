import sympy as sp
# Define the variables
x, y = sp.symbols('x y')

# Define the parameters for Muller-Brown
A = [-200, -100, -170, 15]
a = [-1, -1, -6.5, 0.7]
b = [0, 0, 11, 0.6]
c = [-10, -10, -6.5, 0.7]
x0 = [1, 0, -0.5, -1]
y0 = [0, 0.5, 1.5, 1]

# Define the Muller-Brown function symbolically
f = sum(A[i] * sp.exp(a[i] * (x - x0[i])**2 +
                  b[i] * (x - x0[i]) * (y - y0[i]) +
                  c[i] * (y - y0[i])**2) for i in range(4))

# Compute the gradient (first derivatives)
f_grad = sp.Matrix([f.diff(var) for var in (x, y)])

# Compute the Hessian (second derivatives)
f_hessian = sp.Matrix([[f.diff(var1, var2) for var1 in (x, y)] for var2 in (x, y)])

# Print the results
print("Gradient:\n", f_grad)
print("\nHessian:\n", f_hessian)
print(sp.ccode(f_hessian))
