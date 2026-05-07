from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import pygmsh

#--------------------------------------------------------------------------
import gmsh
import meshio

# ------------------------------
#          MESH GENERATION
# ------------------------------
gmsh.initialize()
gmsh.model.add("rect_with_hole")

# Rectangle: lower left (0,0), width 1.0, height 1.0
gmsh.model.occ.addRectangle(0, 0, 0, 1.0, 1.0, tag=1)

# Circular hole: center (0.5, 0.5), radius 0.2
gmsh.model.occ.addDisk(0.5, 0.5, 0, 0.2, 0.2, tag=2)

# Cut (subtract hole from rectangle)
gmsh.model.occ.cut([(2, 1)], [(2, 2)])

gmsh.model.occ.synchronize()

# Mesh size control
lc = 0.025  # adjust: smaller = finer mesh
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc / 4)

# Generate 2D mesh
gmsh.model.mesh.generate(2)

# Optional: save raw .msh file for inspection
gmsh.write("rect_with_hole.msh")

# Get mesh data
node_tags, coord, _ = gmsh.model.mesh.getNodes()
coord = coord.reshape(-1, 3)[:, :2]  # 2D only

elem_tags, node_tags_elem = gmsh.model.mesh.getElementsByType(2)  # triangles
cells = node_tags_elem.reshape(-1, 3) - 1  # 0-based indexing for DOLFIN

gmsh.finalize()

# Convert to DOLFIN mesh using MeshEditor (legacy style)
mesh = Mesh()
editor = MeshEditor()
editor.open(mesh, "triangle", 2, 2)
editor.init_vertices(len(coord))
editor.init_cells(len(cells))

for i, pt in enumerate(coord):
    editor.add_vertex(i, pt)

for i, cell in enumerate(cells):
    editor.add_cell(i, cell.astype(np.uintp))

editor.close()

print("Mesh created successfully with", mesh.num_cells(), "cells")

#-----------------------------------------------------------------------------------


#Function Spaces
Vu = VectorFunctionSpace(mesh, "CG", 1)   # displacement
Vd = FunctionSpace(mesh,       "CG", 1)   # damage / phase-field
Vs = TensorFunctionSpace(mesh, "DG", 0)   # stress (for output only)
 
u    = Function(Vu, name="Displacement")
d    = Function(Vd, name="Damage")
d_old = Function(Vd, name="Damage_old")   # previous iterate (stagger convergence)

# Material properties
E, nu = 200, 0.2
lmbda  = Constant(E * nu / ((1 + nu) * (1 - 2 * nu)))
mu     = Constant(E / (2 * (1 + nu)))
kappa  = Constant(lmbda + 2.0/3.0 * mu)   # bulk modulus

kres  = Constant(1e-6)          # residual stiffness
Gc    = Constant(1.0)           # critical energy release rate
l0    = Constant(0.02)          # phase-field length scale

# Boundary conditions
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)

class TopBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 1.0) and on_boundary

TopBoundary().mark(boundaries, 1)

ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

def top(x, on_boundary):
    return near(x[1], 1.0) and on_boundary
 
def internal(x, on_boundary):
    return near((x[0] - 0.5)**2 + (x[1] - 0.5)**2, 0.2**2, 0.05) and on_boundary

#disp controlled loading.
Uimp = Expression(("0", "t"), t=0.0, degree=0)
 
bcu = [DirichletBC(Vu, Constant((0, 0)), internal),
       DirichletBC(Vu, Uimp, top)]
 
#Kinematics — Volumetric/Deviatoric Split
def eps(v):
    return sym(grad(v))
 
def tr_pos(A):
    return (tr(A) + abs(tr(A))) / 2.0
 
def tr_neg(A):
    return (tr(A) - abs(tr(A))) / 2.0
 
def psi_plus(v):
    e    = eps(v)
    e_dev = e - (1.0/3.0) * tr(e) * Identity(2)   # deviatoric strain (2D)
    return (kappa / 2.0) * tr_pos(e)**2 + mu * inner(e_dev, e_dev)
 
def psi_minus(v):
    return (kappa / 2.0) * tr_neg(eps(v))**2
 
def degradation(phi):
    return (1.0 - phi)**2 + kres
 
# History Field H
Vh = FunctionSpace(mesh, "DG", 0)
H  = Function(Vh, name="HistoryFunction")
 
def update_history():
    W_plus = project(psi_plus(u), Vh)
    H.vector()[:] = np.maximum(H.vector().get_local(),
                               W_plus.vector().get_local())
 
#Variational Forms
 
# ---- Displacement problem ----
du = TrialFunction(Vu)
vu = TestFunction(Vu)
 
def sigma_degraded(v, phi):
    e = eps(v)
    e_vol = tr(e)
    e_vol_pos = (e_vol + abs(e_vol)) / 2.0
    e_vol_neg = (e_vol - abs(e_vol)) / 2.0
    e_dev = e - (1.0/3.0) * e_vol * Identity(2)

    sig_vol_pos = kappa * e_vol_pos * Identity(2)
    sig_vol_neg = kappa * e_vol_neg * Identity(2)
    sig_dev     = 2.0 * mu * e_dev

    return degradation(phi) * (sig_vol_pos + sig_dev) + sig_vol_neg

# Bilinear and linear forms for u (linear in du for fixed d)
F_u = inner(sigma_degraded(u, d), eps(vu)) * dx

dF_u = derivative(F_u, u, du)

# Nonlinear solver for displacement
problem_u = NonlinearVariationalProblem(F_u, u, bcs=bcu, J=dF_u)
solver_u  = NonlinearVariationalSolver(problem_u)

# Typical parameters (adjust as needed)
prm = solver_u.parameters
prm["newton_solver"]["absolute_tolerance"] = 1e-8
prm["newton_solver"]["relative_tolerance"] = 1e-7
prm["newton_solver"]["maximum_iterations"] = 25
prm["newton_solver"]["linear_solver"]      = "mumps"   # or "lu", "superlu_dist"
prm["newton_solver"]["report"]             = True

dd  = TrialFunction(Vd) 
q   = TestFunction(Vd)
 
def build_damage_forms():
    a = ( (Gc/l0 + 2.0*H) * dd * q
        + Gc * l0 * dot(grad(dd), grad(q)) ) * dx
    L = 2.0 * H * q * dx
    return a, L
 
 
def solve_displacement():
    solver_u.solve()

def solve_damage():
    a_d, L_d = build_damage_forms()
    solve(a_d == L_d, d, solver_parameters={"linear_solver": "lu"})          # or "mumps"
    d.vector()[:] = np.maximum(d.vector().get_local(), d_old.vector().get_local()) # pointwise maximum to enforce irreversibilitya
    d.vector()[:] = np.clip(d.vector().get_local(), 0.0, 1.0) #restricts to [0,1].
 
#Energy functionals
def stored_energy():
    #\int g(d)*W+(u) + W-(u) dx
    return assemble( (degradation(d) * psi_plus(u) + psi_minus(u)) * dx )
 
def dissipated_energy():
    #\int Gc/(2l0)*d^2 + Gc*l0/2*|grad d|^2 dx
    return assemble( (Gc/(2*l0) * d**2 + Gc*l0/2 * dot(grad(d), grad(d))) * dx )
 
#ParaView Output
xdmf_u = XDMFFile("phase_field_no_mfront_displacement.xdmf")
xdmf_d = XDMFFile("phase_field_no_mfront_damage.xdmf")
 
for f in [xdmf_u, xdmf_d]:
    f.parameters["flush_output"]         = True
    f.parameters["functions_share_mesh"] = True
 
#Load-Stepping Loop
tol, Nitermax = 1e-3, 500

loading = np.concatenate((np.linspace(0,   70e-3,  6), np.linspace(70e-3, 125e-3, 26)[1:]))   # skip first zero if you want
N_steps = loading.shape[0]
results = np.zeros((N_steps, 3))   # [force, elastic energy, fracture energy]
 
for i, t in enumerate(loading):
    print("Time step: {}  (u_imp = {:.4f})".format(i+1, t))
    Uimp.t = t
 
    # ---- Alternate minimization ----
    res = 1.0
    j   = 1
    while res > tol and j < Nitermax:
        # Step A: solve mechanics with current d
        solve_displacement()
 
        # Update history field H = max(H, W+(u))
        update_history()
 
        # Step B: solve damage with current u and H
        d_old.assign(d)
        solve_damage()
 
        # Convergence: max pointwise damage increment
        res = np.max(d.vector().get_local() - d_old.vector().get_local())
        print("   Iteration {:3d}:  max(Δd) = {:.2e}".format(j, res))
        j += 1
 
    # ---- Post-processing ----
    n = FacetNormal(mesh)
    traction = dot(sigma_degraded(u, d), n)
    reaction = assemble(dot(traction, as_vector((0,1))) * ds(1)) # top bdry is 1 so ds(1) integrates only there.
    results[i, 0] = reaction
    results[i, 1] = stored_energy()
    results[i, 2] = dissipated_energy()
 
    xdmf_u.write(u, t)
    xdmf_d.write(d, t)
 
    clear_output(wait=True)
    plt.figure()
    p = plot(d, vmin=0, vmax=1)
    plt.colorbar(p)
    plt.title("Damage  t={:.4f}".format(t))
    plt.savefig("./results/phase_field_{:04d}.png".format(i), dpi=400)
    plt.close()
 
xdmf_u.close()
xdmf_d.close()
 
#Summary Plots
plt.figure()
plt.plot(loading, results[:, 0], "-o")
plt.xlabel("Imposed displacement")
plt.ylabel("Vertical force")
plt.title("Load-displacement curve")
plt.show()
 
plt.figure()
plt.plot(loading, results[:, 1], label="elastic energy")
plt.plot(loading, results[:, 2], label="fracture energy")
plt.plot(loading, results[:, 1] + results[:, 2], label="total energy")
plt.xlabel("Imposed displacement")
plt.ylabel("Energies")
plt.legend()
plt.title("Energy evolution")
plt.show()
