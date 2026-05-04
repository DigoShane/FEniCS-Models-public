from fenics import *
import numpy as np

#======================================
# Mesh and function spaces
#======================================
# Create mesh and define function spaces
nx, ny = 40, 40
mesh = UnitSquareMesh(nx, ny)
V_vec = VectorFunctionSpace(mesh, 'CG', 1) #m
V = FunctionSpace(mesh, 'CG', 1) #phi

#======================================
# Functions
#======================================
m = Function(V_vec)
m_old = Function(V_vec)

phi = Function(V)

v = TestFunction(V)
w = TestFunction(V_vec)

# Initialize m
m.interpolate(Constant((1.0, 0.0)))

# =========================================================
# Parameters
# =========================================================
A = Constant(1.0)
K = Constant(1.0)
penalty = Constant(100.0)

# =========================================================
# Helper functions
# =========================================================
def normalize_m(m):
    m_array = m.vector().get_local()
    m_array = m_array.reshape((-1, 2))

    norms = np.linalg.norm(m_array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0

    m_array = m_array / norms
    m.vector()[:] = m_array.flatten()

# =========================================================
# Alternating minimization
# =========================================================
num_iters = 30

for it in range(num_iters):

    print(f"Iteration {it}")

    # -----------------------------------------------------
    # Step 1: Solve Poisson equation for phi
    # Δφ = div(m)
    # -----------------------------------------------------
    F_phi = dot(grad(phi), grad(v))*dx - div(m)*v*dx
    solve(F_phi == 0, phi)

    # -----------------------------------------------------
    # Step 2: Minimize energy w.r.t. m
    # -----------------------------------------------------

    m_trial = TrialFunction(V_vec)

    # Exchange term
    exchange = A * inner(grad(m), grad(w))*dx

    # Anisotropy
    anisotropy = K * (1 - m[0]**2) * w[0]*dx

    # Demag coupling: h_d = -grad(phi)
    h_d = -grad(phi)
    demag = inner(h_d, w)*dx

    # Unit constraint penalty
    penalty_term = penalty * (dot(m, m) - 1) * dot(m, w)*dx

    F_m = exchange + anisotropy + demag + penalty_term

    solve(F_m == 0, m)

    # Normalize (optional but stabilizes)
    normalize_m(m)

# =========================================================
# Compute energy
# =========================================================
energy = assemble(
    A * inner(grad(m), grad(m)) * dx
    + K * (1 - m[0]**2) * dx
    + 0.5 * inner(grad(phi), grad(phi)) * dx
)

print("Final energy:", energy)


from fenics import XDMFFile

xdmf_file = XDMFFile("m_field.xdmf")
xdmf_file.parameters["flush_output"] = True
xdmf_file.parameters["functions_share_mesh"] = True

m.rename("m", "magnetization")
xdmf_file.write(m)

xdmf_file.close()


import matplotlib.pyplot as plt

# ======================================
# Create grid for plotting
# ======================================
n_plot = 100
x_vals = np.linspace(0, 1, n_plot)
y_vals = np.linspace(0, 1, n_plot)

X, Y = np.meshgrid(x_vals, y_vals)

# Storage
m_x = np.zeros_like(X)
m_y = np.zeros_like(Y)

# ======================================
# Evaluate m on grid
# ======================================
for i in range(n_plot):
    for j in range(n_plot):
        point = np.array([X[i, j], Y[i, j]])
        val = m(point)
        m_x[i, j] = val[0]
        m_y[i, j] = val[1]

# ======================================
# Plot m_x
# ======================================
plt.figure()
plt.contourf(X, Y, m_x, levels=50)
plt.colorbar()
plt.title("m_x")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# ======================================
# Plot m_y
# ======================================
plt.figure()
plt.contourf(X, Y, m_y, levels=50)
plt.colorbar()
plt.title("m_y")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# ======================================
# Vector field (very useful)
# ======================================
plt.figure()
plt.quiver(X, Y, m_x, m_y)
plt.title("Magnetization field")
plt.xlabel("x")
plt.ylabel("y")
plt.show()