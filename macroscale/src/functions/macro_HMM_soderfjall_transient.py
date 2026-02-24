import os
from fenics import *
from fenics import parameters

parameters["form_compiler"]["quadrature_degree"] = 8
import numpy as np
from scipy.optimize import root
import numpy as np
import matplotlib.pyplot as plt

set_log_level(LogLevel.ERROR)


class material_parameters:
    def __init__(
        self,
        Rc,
        c,
        rho0,
        eta0,
        t0,
        load_mag,
        load_orientation,
        eccentricity0,
    ):
        self.Rc = Rc
        self.c = c
        self.length_ratio = self.c / self.Rc
        self.rho0 = rho0
        self.eta0 = eta0
        self.t0 = t0
        self.load_mag = load_mag
        self.load_orientation = load_orientation
        self.p0 = 3 * load_mag / (2 * np.pi * Rc**2)
        self.U0 = 12 * eta0 * Rc / (self.p0 * (c**2))
        self.eccentricity0 = np.asarray(eccentricity0) * c / Rc


class solver_parameters:
    def __init__(
        self,
        Rnewton_max_iterations,
        Rnewton_rtol,
        Rnewton_atol,
        Rnewton_relaxation_parameter,
        R_krylov_rtol,
        load_balance_rtol,
        xi,
        bc_tol,
        Id,
        Tend,
        dt,
        t_export_interval,
        angular_velocity_fn,
        dynamic_load_fn,
    ):
        """
        Solver Parameters:
        - Rnewton_max_iterations: int
            Maximum number of iterations for the Newton solver for Reynolds equation
        - Rnewton_rtol: float
            Relative tolerance for the Newton solver for Reynolds equation
        - Rnewton_atol: float
            Absolute tolerance for the Newton solver for Reynolds equation
        - Rnewton_relaxation_parameter: float
            Relaxation parameter for the Newton solver for Reynolds equation
        - R_krylov_rtol: float
            Relative tolerance for the Krylov solver
        - load_balance_rtol: float
            Relative tolerance for the load balancing
        - xi: float
            Penalty parameter
        - bc_tol: float
            Tolerance for identifying nodes on a boundary (|Z|< bc_tol)
        - Id: Identity matrix
        """
        self.Rnewton_max_iterations = Rnewton_max_iterations
        self.Rnewton_rtol = Rnewton_rtol
        self.Rnewton_atol = Rnewton_atol
        self.Rnewton_relaxation_parameter = Rnewton_relaxation_parameter
        self.R_krylov_rtol = R_krylov_rtol
        self.load_balance_rtol = load_balance_rtol
        self.xi = xi
        self.bc_tol = bc_tol
        self.Id = Id
        self.Tend = Tend
        self.dt = dt
        self.t_export_interval = t_export_interval
        self.angular_velocity_fn = angular_velocity_fn
        self.dynamic_load_fn = dynamic_load_fn


class mesh_parameters:
    def __init__(
        self, cupmeshdir, ballmeshdir, CupScale, BallScale, CupDisplace, delta, tol
    ):
        self.cupmeshdir = cupmeshdir
        self.ballmeshdir = ballmeshdir
        self.CupScale = CupScale
        self.BallScale = BallScale
        self.CupDisplace = CupDisplace
        self.delta = delta
        self.tol = tol


class meshfn:
    def __init__(self, mesh_parameters):
        self.mesh_parameters = mesh_parameters
        self.initialise_mesh()

    def initialise_mesh(self):
        meshF = Mesh()
        with XDMFFile(self.mesh_parameters.cupmeshdir) as fileC:
            fileC.read(meshF)
        # Distort and displace cup - Unused and code is not compatible with this any more
        meshF.coordinates()[:, :] *= self.mesh_parameters.CupScale
        meshF.coordinates()[:, :] += self.mesh_parameters.CupDisplace
        marker = MeshFunction("size_t", meshF, meshF.topology().dim() - 1, 0)
        self.marker = marker
        self.meshF = meshF


def boundary(x, tol):
    return near(x[2], 0, tol)


def roelands(p):
    # Parameters (tune to your fluid if needed)
    eta_p = Constant(6.31e-5)  # Pa·s
    eta00 = Constant(1.0e-2)  # Pa·s  (your "MODIFIED" η0)
    p_r = Constant(1.926e9)  # Pa    (Roelands reference pressure; POSITIVE)
    z = Constant(0.548)  # Roelands exponent (often 0.68; you used 0.548)

    # Ensure we don't feed negative pressures (or do what your physics requires):
    p_pos = conditional(gt(p, 0), p, 0.0)

    # Roelands in multiplicative form:  eta = eta_p * (eta00/eta_p)^{(1 + p/p_r)^z}
    base_log = ln(eta00 / eta_p)
    expo = (1.0 + p_pos / p_r) ** z

    # Clamp the exponent to avoid crazy growth
    expo_cap = Constant(10.0)
    expo_c = conditional(lt(expo, expo_cap), expo, expo_cap)

    ln_eta = ln(eta_p) + expo_c * base_log
    return exp(ln_eta)


class HydroSolver:
    def __init__(self, meshin, material_properties, solver_params, output_folder):
        """
        Initialize the HydroSolver with mesh data, material properties, and solver parameters.
        """
        self.output_folder = output_folder
        self.meshin = meshin
        self.cav = "on"
        self.beta = 1
        self.alpha = Constant(1e4)
        self.material_properties = material_properties
        self.solver_params = solver_params
        self.t = 0.0
        self.t_prev = 0.0
        self.Tend = self.solver_params.Tend
        self.dt = self.solver_params.dt
        self.transient_flag = False
        self.setup_function_spaces()
        self.initialise_bc()
        self.initialise_velocity()
        self.update_contact_separation(
            self.material_properties.eccentricity0, HMMState=False, transientState=False
        )
        self.main_residuals = []
        self.init_hprev()
        self.init_rhoprev()
        self.calc_current_load(self.t)
        self.h_file = File("data/output/h_series.pvd")
        self.p_file = File("data/output/p_series.pvd")

    def setup_function_spaces(self):
        """Define Function Spaces and Functions"""

        # Define global normal
        self.global_normal = Expression(("x[0]", "x[1]", "x[2]"), degree=1)
        self.meshin.meshF.init_cell_orientations(self.global_normal)

        self.V = FunctionSpace(self.meshin.meshF, "CG", 1)
        self.W = VectorFunctionSpace(self.meshin.meshF, "CG", 1)
        self.n = self.global_normal

        self.nexp = Expression(
            (
                "n[0]/sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2])",
                "n[1]/sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2])",
                "n[2]/sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2])",
            ),
            n=self.n,
            degree=1,
        )
        self.normal = Function(self.W)
        self.normal.rename("n", "normal")
        self.normal.interpolate(self.nexp)

        self.p = Function(self.V)
        self.v = TestFunction(self.V)
        self.p.rename("p", "pressure")
        self.gradP = Function(self.W)

    def initialise_bc(self):
        self.bc = DirichletBC(
            self.V, 0, lambda x, on_boundary: boundary(x, self.solver_params.bc_tol)
        )

    def initialise_velocity(self):
        self.Uexp = Expression(
            ("x[2]*Rc*U0*omega/2", "0*Rc*U0*omega", "-x[0]*Rc*U0*omega/2"),
            Rc=self.material_properties.Rc,
            t0=self.material_properties.t0,
            U0=self.material_properties.U0,
            omega=self.calc_current_angular_velocity(self.t),
            element=self.W.ufl_element(),
        )
        self.U = interpolate(self.Uexp, self.W)

    def reinitialise_solver(self, eccentricity=None):
        """Refresh solver state for a new step.

        The previously exported rotation matrices are reloaded and the
        function spaces, boundary conditions and lambda are rebuilt.
        If eccentricity is provided, ``material_properties.eccentricity0``
        is updated prior to rebuilding the state.
        """

        if eccentricity is not None:
            self.material_properties.eccentricity0 = np.asarray(eccentricity)

        self.initialise_imported_rotation()
        self.setup_function_spaces()
        self.initialise_bc()
        self.initialise_velocity()

    def calc_current_load(self, t):
        self.current_load = (
            self.solver_params.dynamic_load_fn(t) / self.material_properties.load_mag
        )
        print(f"Current load: {self.current_load}")

    def calc_current_angular_velocity(self, t):
        # print(f'Current omega: {5 * np.pi * np.pi * np.cos(t * 2 * np.pi) / 18}')
        angular_vel = self.solver_params.angular_velocity_fn(t)
        print(f"Current angular velocity: {angular_vel}, time: {t}")
        return angular_vel

    def init_hprev(self):
        self.hprev = Function(self.V)

    def init_rhoprev(self):
        self.rhoprev = Function(self.V)

    def store_hprev(self):
        self.hprev.interpolate(self.h)

    def store_rhoprev(self):
        rho_expr = (self.FluidFraction + self.alpha) / (1 + self.alpha)
        rho_func = project(
            rho_expr, self.V, solver_type="cg", preconditioner_type="hypre_amg"
        )
        self.rhoprev.assign(rho_func)

    def time_step(self):
        self.t_prev = self.t
        self.t += self.dt
        self.calc_current_angular_velocity(self.t)
        self.calc_current_load(self.t)
        self.transient_flag = True

    def set_time(self, T, DT):
        self.t = T
        self.dt = DT
        self.calc_current_angular_velocity(self.t)
        self.calc_current_load(self.t)
        self.transient_flag = True

    def dh_ratio(self, p):
        """ρ/ρ₀ compressibility ratio; returns 1 when gauge-pressure ≤ 0."""
        # Dowson-Higginson compressibility (dimensional)
        C1_DH = 0.59e9
        C2_DH = 1.34
        # IfPos(p, (C1_DH + C2_DH * p) / (C1_DH + p), 1.0)
        return conditional(gt(p, 0), (C1_DH + C2_DH * p) / (C1_DH + p), 1.0)

    def est_pmag_problem(self, HMMState=False, transientState=False):

        eta = 1
        rho = 1

        tgrad_p = grad(self.p) - dot(grad(self.p), self.normal) * self.normal
        tgrad_v = grad(self.v) - dot(grad(self.v), self.normal) * self.normal

        qx_expr = -(self.h**3 / eta) * tgrad_p[0] + self.U[0] * self.h
        qy_expr = -(self.h**3 / eta) * tgrad_p[1] + self.U[1] * self.h
        qz_expr = -(self.h**3 / eta) * tgrad_p[2] + self.U[2] * self.h

        F_initial = (
            (-rho * qx_expr * tgrad_v[0]) * dx
            + (-rho * qy_expr * tgrad_v[1]) * dx
            + (-rho * qz_expr * tgrad_v[2]) * dx
        )

        if transientState:
            dhdt = (self.h - self.hprev) / self.dt
            drhodt = (
                rho - self.rhoprev
            ) / self.dt  # may need to chain rule and turn in to drhodp * dpdt
            F_initial += (rho * dhdt + self.h * drhodt) * self.v * dx

        self.F = F_initial

    def initialise_problem(self, HMMState=False, transientState=False):
        # Currently using an explicit time stepping scheme

        self.FluidFraction = conditional(
            gt(self.p, 0),
            1,
            conditional(
                gt(-self.p - self.beta, 0),
                0,
                1 - 2 * (self.p / self.beta) ** 3 - 3 * (self.p / self.beta) ** 2,
            ),
        )

        # Apply Dowson-Higginson compressibility
        rho = (
            self.dh_ratio(self.p * self.material_properties.p0)
            * (self.FluidFraction + self.alpha)
            / (1 + self.alpha)
        )

        eta_roelands = (
            roelands(self.p * self.material_properties.p0)
            / self.material_properties.eta0
        )

        eta = eta_roelands * (self.FluidFraction + self.alpha) / (1 + self.alpha)
        self.rho = rho

        tgrad_p = grad(self.p) - dot(grad(self.p), self.normal) * self.normal
        tgrad_v = grad(self.v) - dot(grad(self.v), self.normal) * self.normal

        qx_expr = -(self.h**3 / eta) * tgrad_p[0] + self.U[0] * self.h
        qy_expr = -(self.h**3 / eta) * tgrad_p[1] + self.U[1] * self.h
        qz_expr = -(self.h**3 / eta) * tgrad_p[2] + self.U[2] * self.h

        F_initial = (
            (-rho * qx_expr * tgrad_v[0]) * dx
            + (-rho * qy_expr * tgrad_v[1]) * dx
            + (-rho * qz_expr * tgrad_v[2]) * dx
        )

        self.F = F_initial

        if HMMState:
            F_HMM = (
                -rho * self.dQ[0] * tgrad_v[0] * dx
                - rho * self.dQ[1] * tgrad_v[1] * dx
                - rho * self.dQ[2] * tgrad_v[2] * dx
            )
            self.F += F_HMM

        if transientState:
            self.dhdt = (self.h - self.hprev) / self.dt
            self.drhodt = (rho - self.rhoprev) / self.dt
            self.F += (rho * self.dhdt + self.h * self.drhodt) * self.v * dx

    def load_hprev(self, hprev):
        """
        Load the previous contact separation from a np array
        """
        self.hprev.vector()[:] = hprev

    def load_rhoprev(self, p_prev, beta):
        """
        Load the previous fluid fraction from a numpy array
        """
        self.p_prev = Function(self.V)
        self.p_prev.vector()[:] = p_prev

        f = conditional(
            gt(self.p_prev, 0),
            1,
            conditional(
                gt(-self.p_prev - beta, 0),
                0,
                1 - 2 * (self.p_prev / beta) ** 3 - 3 * (self.p_prev / beta) ** 2,
            ),
        )

        rho_expr = (f + 0.001) / (1 + 0.001)
        rho_func = project(
            rho_expr, self.V, solver_type="cg", preconditioner_type="hypre_amg"
        )
        self.rhoprev.assign(rho_func)

    def load_state(self, p, h=None, time=None, dt=None, beta=None):
        """Initialise the solver state from numpy arrays.

        Parameters
        ----------
        p : array_like
            Pressure field to initialise ``self.p``.
        h : array_like, optional
            Previous contact separation used for the transient term.
        time : float, optional
            Current simulation time.
        dt : float, optional
            Current time-step size.
        beta: float, optional
            Beta to use when calculating the previous fluid fraction
        """

        # Set pressure field
        self.p.vector()[:] = p

        # Optional previous contact separation and density
        if h is not None:
            self.load_hprev(h)

        if beta is not None:
            self.load_rhoprev(p, beta)

        # Update time information if provided
        if time is not None or dt is not None:
            T = self.t if time is None else time
            DT = self.dt if dt is None else dt
            self.set_time(T, DT)

    def update_contact_separation(
        self, eccentricity, HMMState=False, transientState=False
    ):
        """
        Update the contact separation based on the given eccentricity.
        """
        sep_expr = Expression(
            ("(1 - (ecc)*sqrt(1 - pow(x[0],2) - pow(x[1],2)))"),
            ecc=eccentricity[2] / self.material_properties.length_ratio,
            element=self.V.ufl_element(),
        )

        self.h = interpolate(sep_expr, self.V)
        h_local = self.h.vector().get_local()
        hmax = np.max(np.nanmax(h_local))
        h_local[np.isnan(h_local)] = hmax
        self.h.vector().set_local(h_local)
        self.h.vector().apply("insert")
        self.h.rename("h", "separation")
        # need to reinitialise the problem to build the new F with updated h.
        self.initialise_problem(HMMState=HMMState, transientState=transientState)

    def update_eccentricity(self, load_balance_err, history, method_params):
        """Update eccentricity using scaling or a secant strategy.

        Parameters
        ----------
        load_balance_err : float
            Current load balance error.
        history : Mapping
            Should provide sequences ``load_balance_errs`` and ``eccentricities``.
            The last value of ``eccentricities`` is assumed to match the current
            eccentricity stored in ``self.material_properties``.
        method_params : Mapping
            Expected to contain ``scaling_factor``.

        Returns
        -------
        tuple[float, str]
            The updated eccentricity (normalised) and the strategy used
            (``"scaling"`` or ``"secant"``).
        """

        scaling = method_params.get("scaling_factor", 0.0)
        ecc_hist = history.get("eccentricities", [])
        err_hist = history.get("load_balance_errs", [])

        current = self.material_properties.eccentricity0[2]

        if len(ecc_hist) < 2 or len(err_hist) < 1:
            new_ecc = current * (1 + load_balance_err * scaling)
            strategy = "scaling"
        else:
            prev_ecc = ecc_hist[-2]
            prev_err = err_hist[-1]
            new_ecc = current - load_balance_err * (current - prev_ecc) / (
                load_balance_err - prev_err
            )
            strategy = "secant"

        self.material_properties.eccentricity0[2] = new_ecc
        return new_ecc, strategy

    def calc_force(self):
        pos_p = conditional(gt(self.p, 0), self.p, 0)
        fx = assemble(-pos_p * self.normal[0] * dx)
        fy = assemble(-pos_p * self.normal[1] * dx)
        fz = assemble(-pos_p * self.normal[2] * dx)

        # Combine components into a single vector and converted to dimensional form
        # Multiplying by 2 to account for the half-space mesh
        self.force = np.array([2 * fx, 2 * fy, 2 * fz])

    def calc_force_pst(self):
        pos_pst = conditional(gt(self.p + self.dP, 0), self.p + self.dP, 0)
        fx = assemble(-(pos_pst) * self.normal[0] * dx)
        fy = assemble(-(pos_pst) * self.normal[1] * dx)
        fz = assemble(-(pos_pst) * self.normal[2] * dx)

        # Combine components into a single vector and converted to dimensional form
        # Multiplying by 2 to account for the half-space mesh
        self.force = np.array([2 * fx, 2 * fy, 2 * fz])

    def calc_shear_stress(self):
        eta = roelands(self.p * self.material_properties.p0)
        F = self.FluidFraction
        eta = F * eta / self.material_properties.eta0
        tau_x = eta * self.U[0] / self.h + 3 * self.h * self.gradP[0]
        tau_y = eta * self.U[1] / self.h + 3 * self.h * self.gradP[1]
        tau_z = eta * self.U[2] / self.h + 3 * self.h * self.gradP[2]
        tau0 = (
            self.material_properties.c
            * self.material_properties.p0
            / (6 * self.material_properties.Rc)
        )
        self.tau = np.array([tau0 * tau_x, tau0 * tau_y, tau0 * tau_z])

    def calc_friction(self):
        # Dimensional
        f_friction_x = self.material_properties.Rc**2 * assemble(self.tau[0] * dx)
        f_friction_y = self.material_properties.Rc**2 * assemble(self.tau[1] * dx)
        f_friction_z = self.material_properties.Rc**2 * assemble(self.tau[2] * dx)

        f_friction = np.array([2 * f_friction_x, 2 * f_friction_y, 2 * f_friction_z])
        print(f"Macro only friction = {f_friction}")
        friction = np.linalg.norm(f_friction[0]) / self.current_load
        self.friction = friction

        self.dim_friction = self.friction
        print(f"Macroscale friction coefficient {self.dim_friction}")

    def calc_hom_friction(self):
        """Calculate friction force using homogenised microscale shear stress."""
        if not hasattr(self, "taust_rot"):
            return
        f_x = self.material_properties.Rc**2 * assemble(self.taust_rot[0] * dx)
        f_y = self.material_properties.Rc**2 * assemble(self.taust_rot[1] * dx)
        f_z = self.material_properties.Rc**2 * assemble(self.taust_rot[2] * dx)
        print(f"Friction {f_x}, {f_y}, {f_z}")
        self.friction_force_hom = np.array([2 * f_x, 0, 2 * f_z])
        self.friction_force_hom_tangential = np.array([2 * f_x, 0, 0])
        self.total_friction_hom = np.linalg.norm(self.friction_force_hom)
        self.tangential_friction_hom = np.linalg.norm(
            self.friction_force_hom_tangential
        )
        self.friction_coeff = self.tangential_friction_hom / self.current_load
        print(f"Homogenised friction coefficient: {self.friction_coeff}")

    def calc_gradP(self):
        tgrad_p = grad(self.p) - dot(grad(self.p), self.normal) * self.normal
        self.gradP = project(tgrad_p, self.W)

    def fenicssolve(self):
        self.J = derivative(self.F, self.p)
        self.problem = NonlinearVariationalProblem(
            self.F, self.p, bcs=self.bc, J=self.J
        )
        self.solver = NonlinearVariationalSolver(self.problem)
        self.solver.parameters["nonlinear_solver"] = "newton"
        newton_solver = self.solver.parameters["newton_solver"]
        newton_solver["maximum_iterations"] = self.solver_params.Rnewton_max_iterations
        newton_solver["relative_tolerance"] = self.solver_params.Rnewton_rtol
        newton_solver["absolute_tolerance"] = self.solver_params.Rnewton_atol
        newton_solver["relaxation_parameter"] = (
            self.solver_params.Rnewton_relaxation_parameter
        )
        newton_solver["krylov_solver"]["nonzero_initial_guess"] = True
        newton_solver["linear_solver"] = "mumps"  # Using GMRES as a krylov solver

        self.solver.solve()
        self.p_new = self.p.copy(deepcopy=True)
        self.calc_force()
        self.calc_shear_stress()
        self.calc_friction()

    def soderfjall_solve(self, HMMState=False, transientState=False):
        alpha_reduction = 0.1
        # find an appropriate beta value by taking a percentage of peak pressure from an uncavitated solve
        self.est_pmag_problem(HMMState=HMMState, transientState=transientState)
        self.fenicssolve()
        p_sol = self.p.vector()[:]
        beta_frac = 0.03
        self.beta = np.max(abs(p_sol)) * beta_frac

        self.alpha.assign(1e4)
        self.initialise_problem(HMMState=HMMState, transientState=transientState)
        while float(self.alpha) > 0.01:
            self.initialise_problem(HMMState=HMMState, transientState=transientState)
            self.fenicssolve()
            self.alpha.assign(float(self.alpha) * alpha_reduction)

        # All the prep for xi for setting up microscale problems
        self.FluidFrac = project(
            self.FluidFraction,
            self.V,
            solver_type="cg",
            preconditioner_type="hypre_amg",
        )
        tgrad_p = grad(self.p) - dot(grad(self.p), self.normal) * self.normal
        self.gradP = project(tgrad_p, self.W)
        p = self.p.vector()[:]
        h = self.h.vector()[:]
        U = self.U.vector()[:]
        U = np.asarray(U).reshape((-1, 3))
        gradp = self.gradP.vector()[:]
        gradp = np.asarray(gradp).reshape((-1, 3))
        _F = self.FluidFrac.vector()[:]
        self.gradF = project(grad(self.FluidFrac), self.W)
        gradFflat = np.asarray(self.gradF.vector()[:]).reshape((-1, 3))

        xi = [
            h,
            p,
            U[0],
            U[1],
            U[2],
            gradp[0],
            gradp[1],
            gradp[2],
            gradFflat[0],
            gradFflat[1],
            gradFflat[2],
        ]
        xi = self.dimensionalise_xi(xi)
        xi.append(_F)

        load = (
            self.current_load
            * np.array(self.material_properties.load_orientation)
            * 2
            * np.pi
            / 3
        )
        self.load = load
        if HMMState:
            print(f"Calculating force with PST correction terms...")
            self.calc_force_pst()
        else:
            self.calc_force()
        difference = load + self.force
        print(
            f"Force: {self.force}, Load: {load}, Difference: {difference[2]}, final alpha: {float(self.alpha)}"
        )
        return xi, difference[2]

    def penalty_residual(self, eccentricity, threshold):
        # return a scaled residual vector based on a threshold distance from contact between ball and cup
        # if the distance is less than the threshold (eccentricity greater than a threshold), return a scaled residual vector
        zero_residual = 3
        ecc_mag = np.linalg.norm(eccentricity)
        res_mag = (
            zero_residual * np.tanh(3 * (ecc_mag - threshold) / (1 - threshold)) + 0.01
        )

        if ecc_mag >= threshold:
            return eccentricity * res_mag / ecc_mag
        else:
            return None

    def test_eccentricity(self, eccentricity, threshold):
        # test if the eccentricity is within the threshold distance from the wall
        if np.linalg.norm(eccentricity) < threshold:
            return True
        return False

    def balance_equation(self, eccentricity, HMMState=False, transientState=False):
        """
        Calculate the difference between the target and actual force vectors.
        """
        length_ratio = self.material_properties.length_ratio
        if (
            self.test_eccentricity(np.asarray(eccentricity) / length_ratio, 0.9999999)
            == False

        ):
            print("eccentricity out of bounds:")
            difference = -self.penalty_residual(eccentricity / length_ratio, 0.9999999)
            print(
                f'Offset: [{" ".join(f"{eccentricity[0] / self.material_properties.length_ratio: >7.2%}")}], '
                f'Difference: [{" ".join(f"{x: >9.2e}" for x in difference)}], '
                f"difference norm: {np.linalg.norm(difference): >9.2e}"
            )
            return difference
        print(f'Start LB with Offset: [{" ".join(f"{eccentricity[0] / self.material_properties.length_ratio: >7.8%}")}] ')
        self.material_properties.eccentricity0 = np.asarray([0, 0, eccentricity[0]])
        self.setup_function_spaces()
        self.initialise_bc()
        self.initialise_velocity()
        self.update_contact_separation(
            self.material_properties.eccentricity0,
            HMMState=HMMState,
            transientState=transientState,
        )
        self.soderfjall_solve(HMMState=HMMState, transientState=transientState)

        load = (
            self.current_load
            * np.array(self.material_properties.load_orientation)
            * 2
            * np.pi
            / 3
        )
        difference = load + self.force
        self.main_residuals.append(np.linalg.norm(difference[2]))

        print(
            f'Offset: [{" ".join(f"{eccentricity[0] / self.material_properties.length_ratio: >3.8%}")}], '
            f'Load: [{" ".join(f"{x: >8.5f}" for x in load)}], '
            f'Force: [{" ".join(f"{x: >8.5f}" for x in self.force)}], '
            f'Difference: [{" ".join(f"{difference[2]: >3.2e}")}], '
            f"difference norm: {np.linalg.norm(difference[2]): >9.4e}"
        )
        print(
            f"Maximum pressure MPa: {self.p.vector().max()*self.material_properties.p0/1e6}"
        )
        return difference[2]

    def solve_loadbalance(self, HMMState=False, transientState=False):
        """
        Use scipy.optimize.root to find the eccentricity that balances the equation.
        """
        initial_guess = self.material_properties.eccentricity0[2]
        self.main_residuals = []
        tol = self.solver_params.load_balance_rtol
        result = root(
            self.balance_equation,
            initial_guess,
            args=(HMMState, transientState),
            method="hybr",
            tol=tol,
        )

        if result.success:
            eccentricity_new = result.x
            self.material_properties.eccentricity0 = np.asarray(
                [0, 0, eccentricity_new[0]]
            )
            print(
                f"Optimization successful: {result.success}, "
                f'Offset: [{" ".join(f"{x / self.material_properties.length_ratio: >7.2%}" for x in self.material_properties.eccentricity0)}]'
            )
            self.setup_function_spaces()
            self.initialise_bc()
            self.initialise_velocity()
            self.update_contact_separation(
                self.material_properties.eccentricity0,
                HMMState=HMMState,
                transientState=transientState,
            )
            self.soderfjall_solve(HMMState=HMMState, transientState=transientState)

            # All the prep for xi for setting up microscale problems
            self.FluidFrac = project(
                self.FluidFraction,
                self.V,
                solver_type="cg",
                preconditioner_type="hypre_amg",
            )
            tgrad_p = grad(self.p) - dot(grad(self.p), self.normal) * self.normal
            self.gradP = project(tgrad_p, self.W)
            p = self.p.vector()[:]
            h = self.h.vector()[:]
            U = self.U.vector()[:]
            U = np.asarray(U).reshape((-1, 3))
            gradp = self.gradP.vector()[:]
            gradp = np.asarray(gradp).reshape((-1, 3))
            _F = self.FluidFrac.vector()[:]
            self.gradF = project(grad(self.FluidFrac), self.W)
            gradFflat = np.asarray(self.gradF.vector()[:]).reshape((-1, 3))

            xi = [
                h,
                p,
                U[0],
                U[1],
                U[2],
                gradp[0],
                gradp[1],
                gradp[2],
                gradFflat[0],
                gradFflat[1],
                gradFflat[2],
            ]
            xi = self.dimensionalise_xi(xi)
            xi.append(_F)
            return xi

        else:
            print("Load balance failed.")
            return None

    def construct_transient_xi(self, xi_new, xi_old):
        h_prev, h_new = xi_old[0, :], xi_new[0, :]
        p_prev, p_new = xi_old[1, :], xi_new[1, :]

        dhdt = self.calc_temporal_gradient(h_new, h_prev)
        dpdt = self.calc_temporal_gradient(p_new, p_prev)
        dhdt = dhdt[np.newaxis, :]
        dpdt = dpdt[np.newaxis, :]

        xi_out = np.concatenate((xi_old, dhdt, dpdt), axis=0)
        return xi_out

    def calc_temporal_gradient(self, new, old):
        # backward difference for calculating temporal gradient
        return (new - old) / self.dt

    def dimensionalise_xi(self, xi):
        """
        Dimensionalise the coupling vector xi.
        """
        U_convt = (
            12
            * self.material_properties.eta0
            * self.material_properties.Rc
            / (self.material_properties.p0 * (self.material_properties.c**2))
        )

        h = xi[0] * self.material_properties.c
        p = xi[1] * self.material_properties.p0
        Ux = xi[2] / U_convt
        Uy = xi[3] / U_convt
        Uz = xi[4] / U_convt
        gradpx = xi[5] * self.material_properties.p0 / self.material_properties.Rc
        gradpy = xi[6] * self.material_properties.p0 / self.material_properties.Rc
        gradpz = xi[7] * self.material_properties.p0 / self.material_properties.Rc
        dfdx = xi[8]
        dfdy = xi[9]
        dfdz = xi[10]

        return [h, p, Ux, Uy, Uz, gradpx, gradpy, gradpz, dfdx, dfdy, dfdz]

    def dimensionalise_xi_transient(self, xi):
        """
        Dimensionalise the coupling vector xi.
        """
        U_convt = (
            12
            * self.material_properties.eta0
            * self.material_properties.Rc
            / (self.material_properties.p0 * (self.material_properties.c**2))
        )

        h = xi[0] * self.material_properties.c
        p = xi[1] * self.material_properties.p0
        Ux = xi[2] / U_convt
        Uy = xi[3] / U_convt
        Uz = xi[4] / U_convt
        gradpx = xi[5] * self.material_properties.p0 / self.material_properties.Lx
        gradpy = xi[6] * self.material_properties.p0 / self.material_properties.Lx
        gradpz = xi[7] * self.material_properties.p0 / self.material_properties.Lx
        dfdx = xi[8]
        dfdy = xi[9]
        dfdz = xi[10]
        dhdt = xi[11] * self.material_properties.c
        drhodt = xi[12] * self.material_properties.rho0

        return [
            h,
            p,
            Ux,
            Uy,
            Uz,
            gradpx,
            gradpy,
            gradpz,
            dfdx,
            dfdy,
            dfdz,
            dhdt,
            drhodt,
        ]

    def calc_gradP(self):
        self.gradP = project(grad(self.p), self.W)

    def export_rotation(self):
        self.rotation_matrix = RotationMatrix(
            self.U, self.normal, self.meshin.meshF, degree=1
        )
        self.rotation_matrix.interpolate()
        self.rotation_inverse = RotationMatrixInverse(
            self.rotation_matrix, self.meshin.meshF, degree=1
        )
        self.rotation_inverse.interpolate()
        self.rotation_matrix.export("rotation_matrix")
        self.rotation_inverse.export("rotation_matrix_inverse")
        self.R = self.rotation_matrix.R_function
        self.R_inv = self.rotation_inverse.R_inv_function

    def initialise_imported_rotation(self):
        self.rotation_matrix = RotationMatrix(
            self.U, self.normal, self.meshin.meshF, degree=1
        )
        self.rotation_matrix.load("rotation_matrix.npy")
        self.rotation_inverse = RotationMatrixInverse(
            self.rotation_matrix, self.meshin.meshF, degree=1
        )
        self.rotation_inverse.load("rotation_matrix_inverse.npy")
        self.R = self.rotation_matrix.R_function
        self.R_inv = self.rotation_inverse.R_inv_function

    def rotate_xi(self):
        R = self.R
        U_rot = rotate_array(self.U, R)
        gradP_rot = rotate_array(self.gradP, R)
        gradF_rot = rotate_array(self.gradF, R)
        xi_r = [
            self.h.vector()[:],
            self.p.vector()[:],
            U_rot[:, 0],
            U_rot[:, 1],
            U_rot[:, 2],
            gradP_rot[:, 0],
            gradP_rot[:, 1],
            gradP_rot[:, 2],
            gradF_rot[:, 0],
            gradF_rot[:, 1],
            gradF_rot[:, 2],
        ]
        xi_r = self.dimensionalise_xi(xi_r)
        xi_r.append(self.FluidFrac.vector()[:])
        return xi_r

    def unrotate_Q(self, dQ):
        R_inv = self.R_inv
        dQ_rot_func = Function(self.W)
        self.dQ = Function(self.W)
        dQ_rot_func.vector()[:] = dQ.flatten()

        self.dQ = rotate_function(dQ_rot_func, R_inv)

        # nondimensionalise the result
        Q0 = (
            self.material_properties.rho0
            * self.material_properties.c**3
            * self.material_properties.p0
            / (12.0 * self.material_properties.eta0 * self.material_properties.Rc)
        )
        self.dQ.vector()[:] = self.dQ.vector()[:] / Q0
        return np.array(self.dQ.vector()[:]).reshape((-1, 3))

    def calcQ(self):
        dh = self.dh_ratio(self.p * self.material_properties.p0)
        # dh = 1
        roelands_eta = (
            roelands(self.p * self.material_properties.p0)
            / self.material_properties.eta0
        )
        Q_exp = self.U * self.h - self.h**3 * self.gradP / (
            (self.FluidFraction + 0.001) / (1 + 0.001) * roelands_eta
        )  ###################Is this right still?
        self.Q = project(Q_exp, self.W)

    def load_dP(self, dP_out):
        self.dP = Function(self.V)
        self.dP.vector()[:] = dP_out.flatten() / self.material_properties.p0

    def load_pmaxmin(self, pmaxin, pminin):
        self.pmax = Function(self.V)
        self.pmax.vector()[:] = pmaxin.flatten() / self.material_properties.p0
        self.pmin = Function(self.V)
        self.pmin.vector()[:] = pminin.flatten() / self.material_properties.p0

    def load_hmaxmin(self, hmaxin, hminin):
        self.hmax = Function(self.V)
        self.hmax.vector()[:] = hmaxin.flatten() / self.material_properties.c
        self.hmin = Function(self.V)
        self.hmin.vector()[:] = hminin.flatten() / self.material_properties.c

    def load_Fst(self, Fst_out):
        self.Fst = Function(self.V)
        self.Fst.vector()[:] = Fst_out.flatten()

    def apply_corrections(
        self,
        dQ,
        taust,
        dP,
        Fst,
        p_bounds=None,
        h_bounds=None,
    ):
        """Load MLS corrections into the solver.

        Parameters
        ----------
        dQ : array-like or sequence of arrays
            Flux corrections.  If given as a sequence, the components are
            stacked and a zero *z* component is appended if necessary.
        taust : array-like or sequence of arrays
            Shear stress corrections.  Behaviour mirrors ``dQ``.
        dP : array-like
            Pressure correction field.
        Fst : array-like
            Fluid fraction correction field.
        p_bounds : tuple(np.ndarray, np.ndarray), optional
            Pair of (pmax, pmin) arrays.
        h_bounds : tuple(np.ndarray, np.ndarray), optional
            Pair of (hmax, hmin) arrays.
        """

        def _stack(arr):
            if isinstance(arr, (tuple, list)):
                arr = np.column_stack(arr)
            else:
                arr = np.asarray(arr)
            if arr.ndim == 1:
                arr = np.column_stack((arr, np.zeros_like(arr), np.zeros_like(arr)))
            elif arr.shape[1] == 2:
                arr = np.column_stack((arr, np.zeros_like(arr[:, 0])))
            return arr

        dQ = _stack(dQ)
        taust = _stack(taust)

        self.unrotate_Q(dQ)
        self.unrotate_taust(taust)
        self.calc_hom_friction()

        self.load_dP(dP)
        self.load_Fst(Fst)

        if p_bounds is not None:
            self.load_pmaxmin(*p_bounds)
        if h_bounds is not None:
            self.load_hmaxmin(*h_bounds)

    def unrotate_taust(self, taust):
        R_inv = self.R_inv
        taust_rot_func = Function(self.W)
        self.taust = Function(self.W)
        taust_rot_func.vector()[:] = taust.flatten()

        # Left in dimensional form since we aren't using it for calculations
        self.taust_rot = rotate_function(taust_rot_func, R_inv)

        return np.array(self.taust.vector()[:]).reshape((-1, 3))

    def export(self, f, tag=None, iter=None, lbiter=None, T=None):
        """
        Export the solution to a file.
        """
        class_name = self.__class__.__name__

        # Create a combined iteration string if both iter and lbiter are provided.
        combined = None
        if T is not None:
            T = T * 10000
            combined = f"{T}"
        elif iter is not None and lbiter is not None and T is None:
            combined = f"{lbiter}-{iter}"
        elif iter is not None and lbiter is not None and T is not None:
            combined = f"{T}-{lbiter}-{iter}"

        if tag is None:
            file_path = f"data/output/{self.output_folder}/{class_name}_{f}.pvd"
        elif combined is None:
            file_path = f"data/output/{self.output_folder}/{class_name}_{f}_{tag}.pvd"
        else:
            file_path = f"data/output/{self.output_folder}/{class_name}_{f}_{tag}_{combined}.pvd"

        file = File(file_path)
        fenics_function = getattr(self, f, None)
        file << fenics_function
        pass

    from pathlib import Path
    from dolfin import XDMFFile, MPI

    def export_series(self, field: str, t: float) -> None:
        """Append a field at physical time t to an XDMF time series.
        Creates the file the first time it is called,
        then re-opens it in append mode on subsequent calls—even if each
        call happens in a new Python process.
        """
        func = getattr(self, field)

        # full path:  data/output/<folder>/HydroSolver_p_transient.xdmf
        path = os.path.join(
            "data",
            "output",
            self.output_folder,
            f"{self.__class__.__name__}_{field}_transient.xdmf",
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # does the XDMF file already exist?
        append = os.path.exists(path)

        with XDMFFile(MPI.comm_world, path) as xdmf:
            xdmf.parameters["functions_share_mesh"] = True
            xdmf.parameters["rewrite_function_mesh"] = False
            xdmf.parameters["flush_output"] = True

            # write the mesh only once, when the file is first created
            if not append:
                xdmf.write(self.meshin.meshF)

            # write the field; 'append' keeps earlier time steps intact
            xdmf.write_checkpoint(
                func,  # the Function to save
                field,  # name shown in ParaView
                float(t),  # physical time
                XDMFFile.Encoding.HDF5,
                append=append,
            )


class RotationMatrix(UserExpression):
    """
    Builds a local orthonormal basis {e1, e2, e3}:
      - e3 is the surface normal at each point.
      - e1 is the (tangential) direction of U at each point,
        after removing any normal component if needed.
      - e2 = e3 x e1.
    Then the rotation matrix R is [ e1^T; e2^T; e3^T ],
    so that R * (global vector) = (local coords).
    """

    def __init__(self, lmb_function, normal_function, mesh, degree=1, **kwargs):
        super().__init__(**kwargs)
        self.lmb_function = lmb_function
        self.normal_function = normal_function
        self.mesh = mesh
        self.R_function = Function(VectorFunctionSpace(mesh, "CG", 1, dim=9))

    def eval(self, values, x):
        lmb = self.lmb_function(x)
        nrm = self.normal_function(x)

        norm_n = np.sqrt(nrm[0] ** 2 + nrm[1] ** 2 + nrm[2] ** 2)
        if norm_n < 1e-18:
            values[:] = np.eye(3).flatten()
            return
        e3 = nrm / norm_n

        dot_lmb_n = lmb[0] * e3[0] + lmb[1] * e3[1] + lmb[2] * e3[2]

        lmb_tan = np.array(
            [
                lmb[0] - dot_lmb_n * e3[0],
                lmb[1] - dot_lmb_n * e3[1],
                lmb[2] - dot_lmb_n * e3[2],
            ]
        )

        norm_tan = np.linalg.norm(lmb_tan)
        if norm_tan < 1e-18:
            e1 = np.array([1.0, 0.0, 0.0])
            cross_test = np.cross(e3, e1)
            if np.linalg.norm(cross_test) < 1e-18:
                e1 = np.array([0.0, 1.0, 0.0])
        else:
            e1 = lmb_tan / norm_tan

        e2 = np.cross(e3, e1)
        norm_e2 = np.linalg.norm(e2)
        if norm_e2 < 1e-18:
            e2 = np.array([0.0, 1.0, 0.0])
            cross_test = np.cross(e3, e2)
            if np.linalg.norm(cross_test) < 1e-18:
                e2 = np.array([0.0, 0.0, 1.0])
        else:
            e2 /= norm_e2

        R = np.array(
            [
                [e1[0], e1[1], e1[2]],
                [e2[0], e2[1], e2[2]],
                [e3[0], e3[1], e3[2]],
            ]
        )
        values[:] = R.flatten()

    def value_shape(self):
        return (9,)

    def interpolate(self):
        self.R_function.interpolate(self)

    def export(self, filename):
        outpath = os.path.join("data/output/", filename)
        np.save(outpath, self.R_function.vector().get_local())

    def load(self, filename):
        inpath = os.path.join("data/output/", filename)
        data = np.load(inpath)
        self.R_function.vector()[:] = data


class RotationMatrixInverse(UserExpression):
    """
    Because the rotation matrix R is orthogonal, its inverse is its transpose.
    This class computes the inverse (or transpose) of the RotationMatrix.
    """

    def __init__(self, rotation_matrix, mesh, degree=1, **kwargs):
        self.rotation_matrix = rotation_matrix
        self.mesh = mesh
        self.R_inv_function = Function(VectorFunctionSpace(mesh, "CG", 1, 9))
        super().__init__(**kwargs)

    def eval(self, values, x):
        # Get the rotation matrix R at point x (flattened to length 9).
        R_flat = self.rotation_matrix(x)
        R = np.array(R_flat).reshape((3, 3))
        # Inverse of an orthogonal matrix is its transpose.
        R_inv = R.T
        values[:] = R_inv.flatten()

    def value_shape(self):
        return (9,)

    def interpolate(self):
        self.R_inv_function.interpolate(self)

    def export(self, filename):
        output_dir = os.path.join("data/output/", filename)
        np.save(output_dir, self.R_inv_function.vector().get_local())

    def load(self, filename):
        input_dir = os.path.join("data/output/", filename)
        data = np.load(input_dir)
        self.R_inv_function.vector()[:] = data


class ApplyRotationMatrix(UserExpression):
    def __init__(self, R_function, Q_function, **kwargs):
        self.R_function = R_function
        self.Q_function = Q_function
        super().__init__(**kwargs)

    def eval(self, values, x):
        R = self.R_function(x).reshape((3, 3))
        Q = self.Q_function(x)
        values[:] = np.dot(R, Q)

    def value_shape(self):
        return (3,)


def rotate_function(func, rotation_matrix):
    rotated_func = Function(func.function_space())
    rotation_expr = ApplyRotationMatrix(rotation_matrix, func, degree=1)
    rotated_func.interpolate(rotation_expr)
    return rotated_func


def rotate_array(func, rotation_matrix):
    rotated_func = Function(func.function_space())
    rotation_expr = ApplyRotationMatrix(rotation_matrix, func, degree=1)
    rotated_func.interpolate(rotation_expr)
    rotated_vals = rotated_func.vector()[:]
    return rotated_vals.reshape((-1, 3))
