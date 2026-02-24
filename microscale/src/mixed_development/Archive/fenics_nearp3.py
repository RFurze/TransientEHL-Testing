from dolfin import *
import numpy as np

# -----------------------------
# Mesh / params
# -----------------------------
n = 100
mesh = UnitSquareMesh(n, n)

rho0 = Constant(1.0)
eta0 = Constant(1.0)

U0 = Constant(1.0)
U  = as_vector((U0, Constant(0.0)))  # slide in +x

# affine periodic jumps for physical pressure P
Ax = Constant(1.0)       # P(0,y) = P(1,y) + Ax
Ay = Constant(-0.5)      # P(x,1) = P(x,0) + Ay

# film thickness (periodic, positive)
x, y = SpatialCoordinate(mesh)
epsH = Constant(0.2)
H = Constant(0.5) + epsH*(cos(2*2*pi*x) + cos(2*2*pi*y))

# -----------------------------
# Periodic BC for w
# -----------------------------
class PeriodicBoundary(SubDomain):
    def inside(self, X, on_boundary):
        return on_boundary and (
            (near(X[0], 0.0) or near(X[1], 0.0)) and
            (not ((near(X[0], 1.0) and near(X[1], 0.0)) or
                  (near(X[0], 0.0) and near(X[1], 1.0)) or
                  (near(X[0], 1.0) and near(X[1], 1.0))))
        )

    def map(self, X, Y):
        if near(X[0], 1.0) and near(X[1], 1.0):
            Y[0] = X[0] - 1.0
            Y[1] = X[1] - 1.0
        elif near(X[0], 1.0):
            Y[0] = X[0] - 1.0
            Y[1] = X[1]
        else:
            Y[0] = X[0]
            Y[1] = X[1] - 1.0

pbc = PeriodicBoundary()
Vper  = FunctionSpace(mesh, "CG", 1, constrained_domain=pbc)
Vfull = FunctionSpace(mesh, "CG", 1)  # for exporting P

# -----------------------------
# Affine ramp phi for jumps
# -----------------------------
phi = Ax*(1.0 - x) + Ay*y   # gives desired jumps when w is periodic

# Unknown in periodic space
w = Function(Vper, name="w")
v = TestFunction(Vper)
dw = TrialFunction(Vper)

# Physical pressure (UFL expression)
P = w + phi

# -----------------------------
# Gauge fixing: pin physical pressure at a point
# P(0,0)=C  =>  w(0,0) = C - phi(0,0) = C - Ax
# -----------------------------
C = Constant(0.5)
bc_pin = DirichletBC(
    Vper,
    C - Ax,
    "near(x[0],0.0) && near(x[1],0.0)",
    method="pointwise"
)

# ============================================================
# 1) UN-CAVITATED SOLVE (rho=eta=1) to compute beta
# ============================================================
K0 = rho0 * H**3 / (12.0 * eta0)
F_adv0 = rho0 * (U * H / 2.0)

R0 = inner(F_adv0 - K0*grad(P), grad(v)) * dx
J0 = derivative(R0, w, dw)

problem0 = NonlinearVariationalProblem(R0, w, bcs=[bc_pin], J=J0)
solver0  = NonlinearVariationalSolver(problem0)

prm0 = solver0.parameters
prm0["newton_solver"]["absolute_tolerance"] = 1e-8
prm0["newton_solver"]["relative_tolerance"] = 1e-6
prm0["newton_solver"]["maximum_iterations"] = 50
prm0["newton_solver"]["report"] = True
prm0["newton_solver"]["error_on_nonconvergence"] = False
prm0["newton_solver"]["relaxation_parameter"] = 0.7
prm0["newton_solver"]["linear_solver"] = "mumps"

w.vector().zero()
solver0.solve()

P0full = project(P, Vfull)

# compute beta = 10% of max pressure in un-cavitated solution
# (sample-based max to avoid needing access to raw dofs in projected space assumptions)
grid = np.linspace(0.0, 1.0, 81)
pmax = -1e100
for xx in grid:
    for yy in grid:
        pmax = max(pmax, P0full(Point(float(xx), float(yy))))

beta_val = 0.1 * pmax
if beta_val <= 0.0:
    # fallback to something positive to avoid division by zero / sign issues
    beta_val = 1e-12

print(f"[UNCAV] approx max(P0)={pmax:.6e} -> beta={beta_val:.6e}")

# ============================================================
# 2) SODERFJALL CAVITATION: alpha-continuation, rho/eta modified
# ============================================================
alpha = Constant(1e4)
beta  = Constant(beta_val)

# dimensionless pressure
pbar = P / beta

# f(pbar) piecewise:
#   1 if pbar > 0
#   polynomial if -0.1 <= pbar <= 0
#   0 if pbar < -0.1
z = pbar / Constant(0.1)
f_poly = 1.0 - 2.0*z**3 - 3.0*z**2

f = conditional(gt(pbar, 0.0),
                Constant(1.0),
                conditional(ge(pbar, -0.1),
                            f_poly,
                            Constant(0.0)))

# s = (f + alpha)/(1 + alpha)
s = (f + alpha) / (1.0 + alpha)

rho_eff = rho0 * s
eta_eff = eta0 * s

K = rho_eff * H**3 / (12.0 * eta_eff)
F_adv = rho_eff * (U * H / 2.0)

R = inner(F_adv - K*grad(P), grad(v)) * dx
J = derivative(R, w, dw)

problem = NonlinearVariationalProblem(R, w, bcs=[bc_pin], J=J)
solver  = NonlinearVariationalSolver(problem)

prm = solver.parameters
prm["newton_solver"]["absolute_tolerance"] = 1e-8
prm["newton_solver"]["relative_tolerance"] = 1e-6
prm["newton_solver"]["maximum_iterations"] = 60
prm["newton_solver"]["report"] = True
prm["newton_solver"]["error_on_nonconvergence"] = False
prm["newton_solver"]["relaxation_parameter"] = 0.2
prm["newton_solver"]["linear_solver"] = "mumps"

# alpha continuation schedule (log-spaced, inclusive)
alpha_vals = np.logspace(4, -2, 5)  # 1e4 -> 1e-2
for a in alpha_vals:
    alpha.assign(float(a))
    print(f"\n[SODERFJALL] Solving with alpha={float(alpha):.6e}, beta={float(beta):.6e}")
    solver.solve()

# -----------------------------
# Export physical pressure
# -----------------------------
Pfull = project(P, Vfull)
Pfull.rename("P", "pressure")
File("pressure_affine_periodic_soderfjall_cav.pvd") << Pfull
print("\nWrote: pressure_affine_periodic_soderfjall_cav.pvd")

hfull = project(H, Vfull)
hfull.rename("H", "film thickness")
File("h_affine_periodic_soderfjall_cav.pvd") << hfull

# -----------------------------
# Diagnostics: check jumps and pin
# -----------------------------
def P_eval(xx, yy):
    return Pfull(Point(float(xx), float(yy)))

xs = np.linspace(0.1, 0.9, 5)
ys = np.linspace(0.1, 0.9, 5)

print("\nCheck top-bottom jump P(x,1)-P(x,0) ~ Ay:")
for xx in xs:
    print(f"x={xx:.2f}: {P_eval(xx,1.0) - P_eval(xx,0.0): .6e} (target {float(Ay): .6e})")

print("\nCheck left-right jump P(0,y)-P(1,y) ~ Ax:")
for yy in ys:
    print(f"y={yy:.2f}: {P_eval(0.0,yy) - P_eval(1.0,yy): .6e} (target {float(Ax): .6e})")

print("Pinned value P(0,0) =", P_eval(0.0, 0.0), " target =", float(C))

# min pressure sample
pmin = 1e100
for xx in np.linspace(0.0, 1.0, 41):
    for yy in np.linspace(0.0, 1.0, 41):
        pmin = min(pmin, P_eval(xx, yy))
print(f"\nApprox min(P) on sampled grid: {pmin:.6e}")
