"""
Shifted periodic BCs with Reynolds
No cavitation
"""

from dolfin import *
import numpy as np

n = 64
mesh = UnitSquareMesh(n, n)

rho = Constant(1.0)
eta = Constant(1.0)

U0 = Constant(1.0)
U  = as_vector((U0, Constant(0.0)))

Ax = Constant(0.5)     # P(0,y) = P(1,y) + Ax
Ay = Constant(-0.2)    # P(x,1) = P(x,0) + Ay

x, y = SpatialCoordinate(mesh)
eps = Constant(0.2)
H = Constant(1.0) + eps*(cos(2*pi*x) + cos(2*pi*y))
K = rho*H**3/(12.0*eta)

# -----------------------------
# Periodic boundary conditions for w
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

# Periodic space for w
Vper  = FunctionSpace(mesh, "CG", 1, constrained_domain=pbc)
# Unconstrained space for physical pressure P
Vfull = FunctionSpace(mesh, "CG", 1)

w = TrialFunction(Vper)
v = TestFunction(Vper)

# Affine ramp (do NOT project this into Vper!)
phi = Ax*(1.0 - x) + Ay*y

F_adv = rho*(U*H/2.0)

a = inner(K*grad(w), grad(v))*dx
L = inner(F_adv, grad(v))*dx - inner(K*grad(phi), grad(v))*dx

# Gauge fixing: enforce P(0,0)=0 => w(0,0) = -phi(0,0) = -Ax
bc_pin = DirichletBC(Vper, Constant(-float(Ax)),
                     "near(x[0],0.0) && near(x[1],0.0)",
                     method="pointwise")

wh = Function(Vper, name="w")
solve(a == L, wh, bcs=[bc_pin])

# Physical pressure in an UNCONSTRAINED space:
# Project the UFL expression wh + phi into Vfull
Ph = project(wh + phi, Vfull)
Ph.rename("P", "pressure")

# -----------------------------
# Checks (now should show the jumps)
# -----------------------------
def P_eval(xx, yy):
    return Ph(Point(float(xx), float(yy)))

xs = np.linspace(0.1, 0.9, 5)
ys = np.linspace(0.1, 0.9, 5)

Ay_val = float(Ay)
Ax_val = float(Ax)

x2 = np.linspace(0.0, 1.0, 20)
y2 = np.linspace(0.0, 1.0, 20)
print("\nPlot along  line x 0.0 to 1, y = 0.5:")
for xx in x2:
    print(f"x={xx:.2f}: {P_eval(xx,0.5)})")
print("\nPlot along  line x 0.5 y = 0 to 1:")
for yy in y2:
    print(f"y={yy:.2f}: {P_eval(0.5,yy)})")

print("\nCheck top-bottom jump P(x,1)-P(x,0) ~ Ay:")
for xx in xs:
    print(f"x={xx:.2f}: {P_eval(xx,1.0) - P_eval(xx,0.0): .6e} (target {Ay_val: .6e})")

print("\nCheck left-right jump P(0,y)-P(1,y) ~ Ax:")
for yy in ys:
    print(f"y={yy:.2f}: {P_eval(0.0,yy) - P_eval(1.0,yy): .6e} (target {Ax_val: .6e})")

File("pressure_affine_periodic.pvd") << Ph
print("\nWrote: pressure_affine_periodic.pvd")



