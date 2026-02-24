import os
from contextlib import contextmanager
from turtle import width
from fenics import *
from fenics import parameters

parameters["form_compiler"]["quadrature_degree"] = 8
import numpy as np
from scipy.optimize import root
from scipy.sparse import load_npz, identity, lil_matrix, csr_matrix
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
        E,
        nu,
        k_spring,
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
        self.E = E
        self.nu = nu
        self.k_spring = k_spring
        denom = 2 * (1 - self.nu**2)
        if self.E <= 0 or denom <= 0:
            self.E_reduced = np.nan
            self.hertz_contact_radius = np.nan
            self.hertz_pmax = np.nan
        else:
            self.E_reduced = self.E / denom
            try:
                self.hertz_contact_radius = (
                    (3 * self.load_mag * self.Rc) / (4 * self.E_reduced)
                ) ** (1 / 3)
                self.hertz_pmax = (
                    3 * self.load_mag / (2 * np.pi * self.hertz_contact_radius**2)
                )
            except ZeroDivisionError:
                self.hertz_contact_radius = np.nan
                self.hertz_pmax = np.nan


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
        K_matrix_dir,
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
        self.K_matrix_dir = K_matrix_dir


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
        meshC = Mesh()
        meshF = Mesh()

        with XDMFFile(self.mesh_parameters.cupmeshdir) as fileC:
            fileC.read(meshC)

        # Distort and displace cup
        meshC.coordinates()[:, :] *= self.mesh_parameters.CupScale
        meshC.coordinates()[:, :] += self.mesh_parameters.CupDisplace

        marker = MeshFunction("size_t", meshC, meshC.topology().dim() - 1, 0)

        Rout_nd = 1.2

        tol_r = 20 * self.mesh_parameters.tol
        inner = InnerSurface(self.mesh_parameters, tol_r=tol_r)
        outer = OuterSurface(self.mesh_parameters, Rout_nd=Rout_nd, tol_r=tol_r)
        cut = CutPlane(self.mesh_parameters, tol_cut=5 * self.mesh_parameters.tol)
        zcut = ZCut(
            self.mesh_parameters, Rout_nd=Rout_nd, tol_cut=20 * self.mesh_parameters.tol
        )

        inner.mark(marker, 1)
        outer.mark(marker, 2)
        zcut.mark(marker, 2)  # fold z=0 plane into outer
        cut.mark(marker, 3)

        submesh = MeshView.create(marker, 1)

        self.marker = marker
        self.meshC = meshC
        self.meshF = submesh


class InnerSurface(SubDomain):
    def __init__(self, mp, tol_r=1e-6):
        super().__init__()
        self.mp = mp
        self.tol_r = tol_r

    def inside(self, x, on_boundary):
        if not on_boundary:
            return False
        # normalize back to the original unit-geometry coordinates
        X = (x[0] - self.mp.CupDisplace[0]) / self.mp.CupScale[0]
        Y = (x[1] - self.mp.CupDisplace[1]) / self.mp.CupScale[1]
        Z = (x[2] - self.mp.CupDisplace[2]) / self.mp.CupScale[2]
        r = (X * X + Y * Y + Z * Z) ** 0.5
        return abs(r - 1.0) <= self.tol_r


class OuterSurface(SubDomain):
    def __init__(self, mp, Rout_nd, tol_r=1e-6):
        super().__init__()
        self.mp = mp
        self.Rout_nd = Rout_nd  # outer_radius / inner_radius in normalized coords
        self.tol_r = tol_r

    def inside(self, x, on_boundary):
        if not on_boundary:
            return False
        X = (x[0] - self.mp.CupDisplace[0]) / self.mp.CupScale[0]
        Y = (x[1] - self.mp.CupDisplace[1]) / self.mp.CupScale[1]
        Z = (x[2] - self.mp.CupDisplace[2]) / self.mp.CupScale[2]
        r = (X * X + Y * Y + Z * Z) ** 0.5
        return abs(r - self.Rout_nd) <= self.tol_r


class CutPlane(SubDomain):
    def __init__(self, mp, tol_cut=None):
        super().__init__()
        self.mp = mp
        self.tol_cut = (5 * mp.tol) if tol_cut is None else tol_cut

    def inside(self, x, on_boundary):
        if not on_boundary:
            return False
        # normalize
        Y = (x[1] - self.mp.CupDisplace[1]) / self.mp.CupScale[1]
        return abs(Y) <= self.tol_cut


class ZCut(SubDomain):
    def __init__(self, mp, Rout_nd, tol_cut=None):
        super().__init__()
        self.mp = mp
        self.Rout_nd = Rout_nd
        self.tol_cut = (5 * mp.tol) if tol_cut is None else tol_cut

    def inside(self, x, on_boundary):
        if not on_boundary:
            return False
        # normalize to unit geometry
        X = (x[0] - self.mp.CupDisplace[0]) / self.mp.CupScale[0]
        Y = (x[1] - self.mp.CupDisplace[1]) / self.mp.CupScale[1]
        Z = (x[2] - self.mp.CupDisplace[2]) / self.mp.CupScale[2]
        # on the flat z=0 cut
        if abs(Z) > self.tol_cut:
            return False
        # radial footprint lies between inner and outer radii (inclusive with tol)
        rho = (X * X + Y * Y) ** 0.5
        return (rho >= 1.0 - self.tol_cut) and (rho <= self.Rout_nd + self.tol_cut)


class ComplementCylinderSubdomain(SubDomain):
    def __init__(self, mesh_parameters):
        self.mesh_parameters = mesh_parameters
        super().__init__()

    def inside(self, x, on_boundary):
        z0 = -1
        # Return True for points not within the specified cylindrical area but on the boundary
        return (
            on_boundary
            and not (
                (x[0] ** 2) / self.mesh_parameters.CupScale[0] ** 2
                + (x[1] ** 2) / self.mesh_parameters.CupScale[1] ** 2
                <= (1 - self.mesh_parameters.tol)
            )
        ) or (
            on_boundary
            and (
                (x[0] ** 2) / self.mesh_parameters.CupScale[0] ** 2
                + (x[1] ** 2) / self.mesh_parameters.CupScale[1] ** 2
                <= (1 - self.mesh_parameters.tol)
            )
            and (x[2] > self.mesh_parameters.CupScale[2] + self.mesh_parameters.tol)
        )


def boundary(x, tol):
    return near(x[2], 0, tol)


def roelands(p):
    eta_p = Constant(6.31e-5)  # Pa·s
    eta00 = Constant(1.0e-2)  # Pa·s
    p_r = Constant(0.5e8)  # Pa 
    z = Constant(0.548)  

    p_pos = conditional(gt(p, 0), p, 0.0)

    # Roelands in multiplicative form:  eta = eta_p * (eta00/eta_p)^{(1 + p/p_r)^z}
    base_log = ln(eta00 / eta_p)
    expo = (1.0 + p_pos / p_r) ** z

    expo_cap = Constant(10.0)
    expo_c = conditional(lt(expo, expo_cap), expo, expo_cap)

    ln_eta = ln(eta_p) + expo_c * base_log
    return exp(ln_eta)


class EHLSolver:
    def __init__(self, meshin, material_properties, solver_params, output_folder):
        """
        Initialize the HydroSolver with mesh data, material properties, and solver parameters.
        """
        self.output_folder = output_folder
        self.meshin = meshin
        self.cav = "on"
        self.beta_value = 1.0
        self.beta = Constant(self.beta_value)
        self.h_regularisation_param = 3e-3  # baseline S before any decay  #NEEDS TUNING? OR DYNAMIC?
        # --- regularisation decay controls/state ---
        self.reg_decay_trigger_factor = (
            50.0  # start decay when p-res <= factor * inner_tol_p
        )
        self.reg_decay_iters = 20  # how many inner iterations to decay over #NEEDS TUNING? OR DYNAMIC?
        self._reg_decay_active = False
        self._reg_decay_start_iter = 0
        self._reg_decay_S0 = float(self.h_regularisation_param)
        self._reg_decay_S_current = float(self.h_regularisation_param)
        self.alpha = Constant(1e4)
        self.material_properties = material_properties
        # --- viscosity relaxation controls/state ---
        self.eta_relax_trigger_factor = 5.0  # pressure-residual factor to start η ramp
        self.eta_relax_iters = 8  # number of inner iterations to reach full η
        self._eta_relax_active = False
        self._eta_relax_start_iter = 0
        self._eta_relax_mixing = 0.0
        self._eta_relax_mixing_param = Constant(self._eta_relax_mixing)
        self._eta_relax_eta0 = Constant(float(self.material_properties.eta0))
        self.solver_params = solver_params
        self.t = 0.0
        self.t_prev = 0.0
        self.Tend = self.solver_params.Tend
        self.dt = self.solver_params.dt
        self.dt_constant = Constant(self.dt)
        self.transient_flag = False
        # ---- regularisation guard defaults ----
        # soft gain/power for residual bump when any nodes are still regularised
        self.reg_guard_gain = 10.0
        self.reg_guard_power = 1.0
        # always add a tiny epsilon so a perfect mechanical balance still doesn't pass if reg. is active
        self.reg_guard_eps = 1e-3
        # if this fraction of nodes were regularised at any time during relaxation, apply a strong bump
        self.reg_guard_reject_frac = 0.02
        # runtime stats container (reset per outer eval)
        self._regularisation_stats = {
            "ever_active": False,
            "last_count": 0,
            "last_fraction": 0.0,
            "max_fraction": 0.0,
            "last_saturated_fraction": 0.0,
            "max_saturated_fraction": 0.0,
            "n_nodes": 0,
        }
        self.load_influence_matrix()
        self.setup_function_spaces()
        self.initialise_bc()
        self.initialise_velocity()
        self.init_hprev()
        self.init_rhoprev()
        self._forms_initialised = False
        self._F_cache = {}
        self.update_contact_separation(
            self.material_properties.eccentricity0,
            HMMState=False,
            transientState=False,
            EHLState=False,
        )
        self.main_residuals = []
        self.calc_current_load(self.t)
        if np.isfinite(self.material_properties.hertz_pmax):
            print(
                "Hertz reference: a = "
                f"{self.material_properties.hertz_contact_radius:.6e} m, p_max = "
                f"{self.material_properties.hertz_pmax/1e6:.3f} MPa"
            )
        self.h_file = File("data/output/h_series.pvd")
        self.p_file = File("data/output/p_series.pvd")
        self._balance_cache: dict[tuple[float, bool, bool], tuple] = {}
        self._memo_count = 0
        self._cache_suppressed = 0
        # ---- inner-loop (EHL relaxation) quick-win defaults ----
        # tolerances are non-dimensional (consistent with p, δ fields)
        self.inner_tol_p = 1e-5  # pressure mixing residual target
        self.inner_tol_d = 1e-5  # deformation mixing residual target
        self.inner_min_iters = (
            8  # do at least this many inner iterations before testing tol
        )
        self._skip_function_space_setup = False

    # --- helper: current smoothing S for h-regularisation ---
    def _get_h_smoothing(self) -> float:
        try:
            return float(self._reg_decay_S_current)
        except Exception:
            return float(self.h_regularisation_param)

    # --- helper: (re)initialise smoothing decay state for a new outer solve ---
    def _reset_reg_decay(self) -> None:
        self._reg_decay_active = False
        self._reg_decay_start_iter = 0
        self._reg_decay_S0 = float(self.h_regularisation_param)
        self._reg_decay_S_current = float(self.h_regularisation_param)

    def _reset_eta_relaxation(self) -> None:
        self._eta_relax_active = False
        self._eta_relax_start_iter = 0
        self._eta_relax_mixing = 0.0
        self._eta_relax_mixing_param.assign(self._eta_relax_mixing)

    def _eta_relaxation_progress(self) -> float:
        return float(self._eta_relax_mixing)

    def _eta_relaxation_complete(self) -> bool:
        return self._eta_relaxation_progress() >= 1.0 - 1e-8

    def _update_eta_relaxation(
        self, pressure_residual: float, relaxation_iteration: int, trigger: float
    ) -> float:
        if not self._eta_relax_active and pressure_residual <= trigger:
            self._eta_relax_active = True
            self._eta_relax_start_iter = relaxation_iteration
            print(
                f"[ETA-RELAX] Starting viscosity ramp: trigger={trigger:.3e}, "
                f"iter={relaxation_iteration}"
            )

        if self._eta_relax_active and self._eta_relax_mixing < 1.0:
            steps = max(int(self.eta_relax_iters), 1)
            prog = max(
                0.0,
                min(
                    1.0, (relaxation_iteration - self._eta_relax_start_iter + 1) / steps
                ),
            )
            if prog > self._eta_relax_mixing + 1e-12:
                self._eta_relax_mixing = prog
                self._eta_relax_mixing_param.assign(self._eta_relax_mixing)
                print(
                    f"[ETA-RELAX] iter={relaxation_iteration} "
                    f"mix={self._eta_relax_mixing:.3f}"
                )

        return self._eta_relax_mixing

    def _roelands_effective(self, p):
        eta_true = roelands(p)
        return (
            self._eta_relax_eta0
            + (eta_true - self._eta_relax_eta0) * self._eta_relax_mixing_param
        )

    @contextmanager
    def _suspend_balance_cache(self):
        """Temporarily disable reading/writing the balance cache."""
        self._cache_suppressed += 1
        try:
            yield
        finally:
            self._cache_suppressed -= 1

    def setup_function_spaces(self):
        """Define Function Spaces and Functions"""

        # Define global normal
        self.global_normal = Expression(("x[0]", "x[1]", "x[2]"), degree=1)
        self.meshin.meshF.init_cell_orientations(self.global_normal)

        self.V = FunctionSpace(self.meshin.meshF, "CG", 1)
        self.W = VectorFunctionSpace(self.meshin.meshF, "CG", 1)
        # cache node count for regularisation statistics
        try:
            self._regularisation_stats["n_nodes"] = int(self.V.dim())
        except Exception:
            self._regularisation_stats["n_nodes"] = 0

        self.n = self.global_normal

        if hasattr(self, "_F_cache"):
            self._F_cache.clear()
        self._forms_initialised = False

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
        self.PRelax = Function(self.V)
        self.Pnow = Function(self.V)
        self.v = TestFunction(self.V)
        self.p.rename("p", "pressure")
        self.gradP = Function(self.W)

        self.delta = Function(self.V)
        self.p_old = Function(
            self.V
        )  # previous pressure for calculating residual during EHL deformation loop
        self.h = Function(self.V)
        self.h.rename("h", "separation")
        self.dQ = Function(self.W)

    def initialise_bc(self):
        self.bc = DirichletBC(
            self.V, 0, lambda x, on_boundary: boundary(x, self.solver_params.bc_tol)
        )

    def initialise_velocity(self):
        self.Uexp = Expression(
            ("x[2]*Rc*U0*omega", "0*Rc*U0*omega", "-x[0]*Rc*U0*omega"),
            Rc=self.material_properties.Rc,
            t0=self.material_properties.t0,
            U0=self.material_properties.U0,
            omega=self.calc_current_angular_velocity(self.t),
            element=self.W.ufl_element(),
        )
        self.U = interpolate(self.Uexp, self.W)

    def load_influence_matrix(self):
        self.K = load_npz(self.solver_params.K_matrix_dir)
        # Convert K to a CSR matrix
        self.K = self.K.tocsr()
        print("Loaded influence matrix")

    def calculate_deformation(self, p, K):
        """Calculate the elastic deformation from the pressure field using the influence matrix."""
        # Input: p as a fenics function, K as a scipy sparse matrix
        # Output: delta as an array, to be loaded in to self.delta or other later
        # The method can be used for any K and p
        ######################################################### NEED TO REVISIT TO CHECK DIMENSIONAL/NONDIMENSIONAL CONSISTENCY
        p_array = p.vector()[:]
        # remove negative pressures from the array
        p_array = np.maximum(p_array, 0)
        delta = K @ p_array * self.material_properties.p0 / (1e6)
        # print(f'Max deformation: {np.max(delta)/self.material_properties.c} m')
        return delta / self.material_properties.c

    def reinitialise_solver(self, eccentricity=None):
        """Refresh solver state for a new step.

        The previously exported rotation matrices are reloaded and the
        function spaces, boundary conditions and lambda are rebuilt.
        If eccentricity is provided, ``material_properties.eccentricity0``
        is updated prior to rebuilding the state.
        """

        if eccentricity is not None:
            self.material_properties.eccentricity0 = np.asarray(eccentricity)

        self._skip_function_space_setup = False
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
        self.dt_constant.assign(self.dt)
        self.calc_current_angular_velocity(self.t)
        self.calc_current_load(self.t)
        self.transient_flag = True

    def dh_ratio(self, p):
        """ρ/ρ₀ compressibility ratio; returns 1 when gauge-pressure ≤ 0."""
        # Dowson-Higginson compressibility (dimensional)
        C1_DH = 0.59e9
        C2_DH = 1.34
        # IfPos(p, (C1_DH + C2_DH * p) / (C1_DH + p), 1.0)
        return 1  # conditional(gt(p, 0), (C1_DH + C2_DH * p) / (C1_DH + p), 1.0)

    def set_beta(self, value: float) -> None:
        self.beta_value = float(value/self.material_properties.p0)
        if hasattr(self.beta, "assign"):
            self.beta.assign(self.beta_value)
        else:
            # Fallback in case self.beta was replaced by a raw float (e.g. by
            # serialisation/initialisation paths). Recreate the Constant so
            # downstream UFL operations continue to work as expected.
            self.beta = Constant(self.beta_value)

    def _setup_forms(self):
        if self._forms_initialised:
            return

        if not hasattr(self, "hprev"):
            print(f"Initialising hprev")
            self.init_hprev()
        if not hasattr(self, "rhoprev"):
            self.init_rhoprev()

        self.FluidFraction = conditional(
            gt(self.p, 0),
            1,
            conditional(
                gt(-self.p - self.beta, 0),
                0,
                1 - 2 * (self.p / self.beta) ** 3 - 3 * (self.p / self.beta) ** 2,
            ),
        )

        rho_expr = (
            self.dh_ratio(self.p * self.material_properties.p0)
            * (self.FluidFraction + self.alpha)
            / (1 + self.alpha)
        )

        eta_roelands = (
            self._roelands_effective(self.p * self.material_properties.p0)
            / self.material_properties.eta0
        )

        eta_expr = eta_roelands * (self.FluidFraction + self.alpha) / (1 + self.alpha)

        tgrad_p = grad(self.p) - dot(grad(self.p), self.normal) * self.normal
        tgrad_v = grad(self.v) - dot(grad(self.v), self.normal) * self.normal

        qx_expr = -(self.h**3 / eta_expr) * tgrad_p[0] + self.U[0] * self.h
        qy_expr = -(self.h**3 / eta_expr) * tgrad_p[1] + self.U[1] * self.h
        qz_expr = -(self.h**3 / eta_expr) * tgrad_p[2] + self.U[2] * self.h

        self._rho_expr = rho_expr
        self._eta_expr = eta_expr
        self._tgrad_p = tgrad_p
        self._tgrad_v = tgrad_v

        self._F_base = (
            (-rho_expr * qx_expr * tgrad_v[0]) * dx
            + (-rho_expr * qy_expr * tgrad_v[1]) * dx
            + (-rho_expr * qz_expr * tgrad_v[2]) * dx
        )

        self._F_hmm = (
            -rho_expr * self.dQ[0] * tgrad_v[0] * dx
            - rho_expr * self.dQ[1] * tgrad_v[1] * dx
            - rho_expr * self.dQ[2] * tgrad_v[2] * dx
        )

        dhdt_expr = (self.h - self.hprev) / self.dt_constant
        drhodt_expr = (rho_expr - self.rhoprev) / self.dt_constant

        self._F_transient = (rho_expr * dhdt_expr + self.h * drhodt_expr) * self.v * dx

        self._forms_initialised = True
        self._F_cache.clear()

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

        self._setup_forms()

        key = (HMMState, transientState)
        if key not in self._F_cache:
            F_form = self._F_base
            if HMMState:
                F_form += self._F_hmm
            if transientState:
                F_form += self._F_transient
            self._F_cache[key] = F_form

        self.F = self._F_cache[key]

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

    def load_state(self, p, deform, h=None, time=None, dt=None, beta=None):
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

        # Set pressure field and deformations
        print(f"Loading state with max p = {np.max(p):.3e}")
        self.p.vector()[:] = p
        self.delta.vector()[:] = deform
        print(f'Maximum deformation: {np.max(deform)}')

        # Optional previous contact separation and density
        if h is not None:
            print(f"Loading hprev with min h = {np.min(h):.3e}")
            self.load_hprev(h)

        if beta is not None:
            # beta is stored/loaded as dimensional (Pa). set_beta converts it to ND.
            self.set_beta(beta)
            # load_rhoprev expects beta in the SAME units as p_prev (non-dimensional)
            self.load_rhoprev(p, float(self.beta))            

        # Update time information if provided
        if time is not None or dt is not None:
            T = self.t if time is None else time
            DT = self.dt if dt is None else dt
            self.set_time(T, DT)
        
        #Once state is loaded we need to prevent reinitialising function spaces and overwriting the loaded data
        self._skip_function_space_setup = True

    def regularise_contact_separation(
        self,
        values: np.ndarray,
        smoothing: float | None = None,
        h_min: float | None = None,
    ) -> np.ndarray:
        """
        Smooth floor regularisation of film thickness. A node is counted as
        'regularised' if the regularised value differs by > 0.1% from
        (unregularised film + current deformation at that node).

        If the caller has already included deformation in `values` (EHLState=True),
        we avoid double-counting δ by using a small context flag set in
        `update_contact_separation`.
        """

        if values.size == 0:
            return values

        # Defaults
        if smoothing is None:
            smoothing = float(self.h_regularisation_param)
        if h_min is None:
            h_min = float(getattr(self, "h_min_regularisation", 0.0))

        # Work on a copy; keep the original (unregularised) values as 'orig'
        vals = values.astype(float, copy=True)
        orig = vals.copy()

        # Replace non-finite entries
        finite_mask = np.isfinite(vals)
        if not np.all(finite_mask):
            fill_value = vals[finite_mask].max() if np.any(finite_mask) else 0.0
            vals[~finite_mask] = fill_value
            orig[~finite_mask] = fill_value  # keep reference consistent too

        # Smooth max relative to floor
        x = vals - h_min
        vals_reg = h_min + 0.5 * (x + np.hypot(x, smoothing))
        np.maximum(vals_reg, h_min, out=vals_reg)

        # ----- Build the reference: (unregularised film + current δ) -----
        # If values already included δ (EHLState=True), don't add δ again.
        ctx = getattr(self, "_regularisation_reference", "geom_only")
        try:
            delta_total = self.delta.vector()[:]
        except Exception:
            # If δ is unavailable, fall back to zeros
            delta_total = 0.0

        if ctx == "with_delta":
            reference = orig  # already contains δ
        else:
            # geometric-only incoming values: add current δ to form the reference
            reference = orig + delta_total

        # ----- Classification: relative deviation > 1e-3 (0.1%) -----
        eps = 1e-15
        relative_diff = np.abs(vals_reg - reference) / (np.abs(reference) + eps)
        regularised_mask = relative_diff > 1e-3
        # print(f'Largest relative diff after reg: {relative_diff.max():.3e}')
        original_count = int(np.count_nonzero(regularised_mask))

        # ----- Stats for guard/diagnostics (unchanged semantics) -----
        try:
            n_nodes = int(
                self._regularisation_stats.get("n_nodes", len(vals_reg))
            ) or len(vals_reg)

            # "Touched" nodes (absolute uplift diagnostic)
            uplift = vals_reg - orig
            uplift_tol = max(1e-14, 1e-6 * smoothing)
            touched = np.count_nonzero(uplift > uplift_tol)
            touched_frac = (touched / n_nodes) if n_nodes > 0 else 0.0

            # Near-floor saturation (for REG-GUARD)
            sat_tol = max(1e-12, 0.1 * smoothing)
            saturated = np.count_nonzero(vals_reg <= (h_min + sat_tol))
            saturated_frac = (saturated / n_nodes) if n_nodes > 0 else 0.0

            frac = (original_count / n_nodes) if n_nodes > 0 else 0.0
            ever = (original_count > 0) or (saturated > 0)

            self._regularisation_stats["ever_active"] = (
                self._regularisation_stats.get("ever_active", False) or ever
            )
            self._regularisation_stats["last_count"] = original_count
            self._regularisation_stats["last_fraction"] = float(frac)
            self._regularisation_stats["max_fraction"] = max(
                self._regularisation_stats.get("max_fraction", 0.0), float(frac)
            )
            self._regularisation_stats["last_saturated_fraction"] = float(
                saturated_frac
            )
            self._regularisation_stats["max_saturated_fraction"] = max(
                self._regularisation_stats.get("max_saturated_fraction", 0.0),
                float(saturated_frac),
            )
            self._regularisation_stats["last_touched_fraction"] = float(touched_frac)
        except Exception:
            pass

        # if original_count:
        #     print(
        #         f"Regularised {original_count} film nodes (>0.1% relative change w.r.t. h_unreg + δ) "
        #         f"with smoothing {smoothing:.3e}; h_min={h_min:.3e}."
        #     )

        return vals_reg

    def update_contact_separation(
        self, eccentricity, HMMState=False, transientState=False, EHLState=False
    ):
        """
        Update the contact separation based on the given eccentricity.
        """
        if EHLState:
            if not hasattr(self, "_sep_expr_ehl"):
                self._sep_expr_ehl = Expression(
                    (
                        "(1 - (ecc)*sqrt(1 - pow(x[0],2) - pow(x[1],2))) + "
                        "delta_total"
                    ),
                    ecc=0.0,
                    delta_total=self.delta,
                    element=self.V.ufl_element(),
                )
            self._sep_expr_ehl.ecc = (
                eccentricity[2] / self.material_properties.length_ratio
            )
            sep_func = interpolate(self._sep_expr_ehl, self.V)
            self._regularisation_reference = "with_delta"
        else:
            if not hasattr(self, "_sep_expr"):
                self._sep_expr = Expression(
                    ("(1 - (ecc)*sqrt(1 - pow(x[0],2) - pow(x[1],2)))"),
                    ecc=0.0,
                    element=self.V.ufl_element(),
                )
            self._sep_expr.ecc = eccentricity[2] / self.material_properties.length_ratio
            sep_func = interpolate(self._sep_expr, self.V)
            self._regularisation_reference = "geom_only"

        h_local = sep_func.vector().get_local()
        # Use the *current* smoothing S (may be decaying to 0)
        h_local = self.regularise_contact_separation(
            h_local, smoothing=self._get_h_smoothing(), h_min=0.0
        )
        self.h.vector().set_local(h_local)
        self.h.vector().apply("insert")
        # need to reinitialise the problem to build the new F with updated h.
        self.initialise_problem(HMMState=HMMState, transientState=transientState)

        # keep max fraction up to date if called multiple times in a loop
        try:
            self._regularisation_stats["max_saturated_fraction"] = max(
                self._regularisation_stats.get("max_saturated_fraction", 0.0),
                self._regularisation_stats.get("last_saturated_fraction", 0.0),
            )
        except Exception:
            pass

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
        eta = self._roelands_effective(self.p * self.material_properties.p0)
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
        friction = np.linalg.norm(f_friction[0]) / (
            self.current_load * self.material_properties.load_mag
        )
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
        # print(f"Friction {f_x}, {f_y}, {f_z}")
        self.friction_force_hom = np.array([2 * f_x, 0, 2 * f_z])
        self.friction_force_hom_tangential = np.array([2 * f_x, 0, 0])
        self.total_friction_hom = np.linalg.norm(self.friction_force_hom)
        self.tangential_friction_hom = np.linalg.norm(
            self.friction_force_hom_tangential
        )
        self.friction_coeff = self.tangential_friction_hom / (
            self.current_load * self.material_properties.load_mag
        )
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

    def soderfjall_solve(
        self,
        HMMState=False,
        transientState=False,
        *,
        postprocess=True,
        full=True,
    ):
        """Solve the macroscale Soderfjäll problem.

        Parameters
        ----------
        HMMState, transientState
            Flags controlling the type of solve to perform (unchanged).
        postprocess
            When ``True`` (the default) the expensive projections required to
            build the microscale ``xi`` fields are executed.  Setting this to
            ``False`` keeps the pressure solve identical but skips the
            post-processing, which is useful inside the EHL relaxation loop
            where only the updated pressure/force is required.
        full
            When ``False``, **skip** beta estimation and alpha continuation,
            reusing current ``beta``/``alpha`` and performing **one** Newton
            solve only. Use this inside relaxation iterations > 0.
        """
        alpha_reduction = 0.1
        if not full:
            # --- fast path: single Newton with current beta/alpha, no re-init ---
            try:
                _ = self.F
            except AttributeError:
                self.initialise_problem(
                    HMMState=HMMState, transientState=transientState
                )
            # --- stable beta update (with freeze-on-close) ---
            if not getattr(self, "_beta_freeze_active", False):
                beta_frac = 0.01
                p_peak = float(np.max(np.abs(self.p.vector()[:])))
                target_beta = beta_frac * p_peak * self.material_properties.p0

                # stateful EMA storage
                if not hasattr(self, "_beta_ema"):
                    self._beta_ema = float(self.beta)

                # optional: only update every k inner iterations to reduce jitter
                k = 3
                if (getattr(self, "_relax_k_counter", 0) % k) == 0:
                    # gate increases if last pressure residual got worse
                    allow_increase = True
                    if hasattr(self, "prev_pressure_residual"):
                        allow_increase = (
                            self.prev_pressure_residual is None
                        ) or np.isfinite(self.prev_pressure_residual)

                    # EMA smoothing
                    ema_alpha = 0.2  # smaller = smoother beta
                    proposed = (
                        1.0 - ema_alpha
                    ) * self._beta_ema + ema_alpha * target_beta

                    # if last p-res increased, only allow decreasing beta (less aggressive cavitation)
                    if not allow_increase and proposed > self._beta_ema:
                        proposed = self._beta_ema

                    print(f"[BETA] New beta {proposed:.6g}, old {float(self.beta):.6g}, alpha = {self.alpha}")
                    self._beta_ema = proposed
                    self.set_beta(proposed)
            else:
                # beta frozen by inner loop; do not update
                if getattr(self, "_beta_freeze_just_set", False):
                    # print(f"[BETA] Frozen at {float(self.beta):.6g}")
                    self._beta_freeze_just_set = False

            self._relax_k_counter = getattr(self, "_relax_k_counter", 0) + 1
            # print(
            #     f"Current Beta: {float(self.beta):.6g}, beta freeze active: {self._beta_freeze_active}, full {full}"
            # )
            self.fenicssolve()
        else:
            # --- original full path: beta estimate + alpha continuation ---
            # find an appropriate beta value by taking a percentage of peak pressure from an uncavitated solve
            self.est_pmag_problem(HMMState=HMMState, transientState=transientState)
            self.fenicssolve()
            p_sol = self.p.vector()[:]
            beta_frac = 0.01
            self.set_beta(np.max(abs(p_sol)) * beta_frac * self.material_properties.p0)

            self.alpha.assign(1e4)
            self.initialise_problem(HMMState=HMMState, transientState=transientState)
            while float(self.alpha) > 0.01:
                self.fenicssolve()
                self.alpha.assign(float(self.alpha) * alpha_reduction)

        load = (
            self.current_load
            * np.array(self.material_properties.load_orientation)
            * 2
            * np.pi
            / 3
        )
        self.load = load
        if HMMState:
            # print(f"Calculating force with PST correction terms...")
            self.calc_force_pst()
        else:
            self.calc_force()
        difference = load + self.force

        # print(f"Force: {self.force}, Load: {load}, Difference: {difference[2]}, final alpha: {float(self.alpha)}")

        if not postprocess:
            return None, difference[2]

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
        h = h - self.local_deformation()
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
        key = (eccentricity, HMMState, transientState)
        if key in self._balance_cache:
            self._memo_count += 1
            return self._balance_cache[key]
        print(f"Solving with eccentricity {eccentricity}")
        self.material_properties.eccentricity0 = np.asarray([0, 0, eccentricity])
        self.setup_function_spaces()
        self.initialise_bc()
        self.initialise_velocity()
        self.update_contact_separation(
            self.material_properties.eccentricity0,
            HMMState=HMMState,
            transientState=transientState,
            EHLState=False,
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
            f"Offset: [{self.material_properties.eccentricity0[2] / self.material_properties.length_ratio: >3.8%}], "
            f'Load: [{" ".join(f"{x: >8.5f}" for x in load)}], '
            f'Force: [{" ".join(f"{x: >8.5f}" for x in self.force)}], '
            f"Difference: [{difference[2]: >3.2e}], "
            f"difference norm: {np.linalg.norm(difference[2]): >9.4e}"
        )
        print(
            f"Maximum pressure MPa: {self.p.vector().max()*self.material_properties.p0/1e6}"
        )
        print(
            f"Minimum film thickness mm: {self.h.vector().min()*self.material_properties.c}"
        )
        print(f"final alpha: {float(self.alpha)}")
        print(f"beta value {float(self.beta)}")
        self._balance_cache[key] = (None, difference[2])
        return (None, difference[2])

    def relative_residual(self, new, old):
        if np.linalg.norm(old) < 1e-15:
            return np.linalg.norm(new - old)
        return np.linalg.norm(new - old) / np.linalg.norm(old)

    def _as_scalar(self, x) -> float:
        """Accept float or ndarray-like, return plain float (first element)."""
        return float(np.asarray(x).reshape(-1)[0])

    def _cache_key(self, ecc, HMMState: bool, transientState: bool):
        """Build a hashable, stable key (rounded to avoid fp noise)."""
        return (round(self._as_scalar(ecc), 12), bool(HMMState), bool(transientState))

    def _smooth_penalty(
        self,
        frac: float,
        *,
        f0: float = 0.005,
        width: float = 0.01,
        gain: float = 5.0,
        eps: float = 1e-4,
    ) -> float:
        """
        Smooth penalty in [eps, ~gain+eps] that turns on continuously once the
        regularised-node fraction 'frac' exceeds f0, with transition width.
        """
        s = 0.5 * (1.0 + np.tanh((float(frac) - f0) / max(width, 1e-6)))
        return gain * (s * s) + eps

    def _apply_regularisation_guard(self, resid_nd: float) -> float:
        """
        Apply a smooth, sign-preserving penalty to the *normalised* residual
        only if the floor regulariser is still active at the end of relaxation.
        """
        stats = self._regularisation_stats or {}
        # Use the *end-of-loop* fractions only
        frac_sat_end = float(stats.get("last_saturated_fraction", 0.0) or 0.0)
        frac_pre_end = float(stats.get("last_fraction", 0.0) or 0.0)
        frac_end = max(frac_sat_end, frac_pre_end)
        if frac_end <= 0.0:
            # No regularisation active in the final state → no penalty
            return resid_nd
        sgn = 1.0 if resid_nd >= 0.0 else -1.0
        bump = self._smooth_penalty(frac_end)
        print(
            f"[REG-GUARD] active at end: pre_end={frac_pre_end*100:.2f}%, "
            f"sat_end={frac_sat_end*100:.2f}% → +{bump:.3e} (normalised)"
        )
        return sgn * (abs(resid_nd) + bump)

    def EHL_HMM_solve(self, eccentricity, HMMState=True, transientState=False):
        """
        HMM macro solve with *no dP/dQ relaxation*.
        Iterate: (re)build δ from current (p, dP) → update h → single Newton on p,
        until the **pressure residual** is below `inner_tol_p`. Keep alpha and beta fixed.
        """
        ecc = self._as_scalar(eccentricity)
        key = self._cache_key(ecc, HMMState, transientState)

        if not self._cache_suppressed and key in self._balance_cache:
            self._memo_count += 1
            cached = self._balance_cache[key]
            if not isinstance(cached, (tuple, list)) or len(cached) != 2:
                return (None, float(cached))
            return cached

        print(f"Solving with eccentricity {eccentricity}")
        # Freeze beta inside soderfjall_solve(full=False); do not touch alpha.
        self._beta_freeze_active = True
        self._beta_freeze_just_set = False
        self._regularisation_stats = {
            "ever_active": False,
            "last_count": 0,
            "last_fraction": 0.0,
            "max_fraction": 0.0,
            "last_saturated_fraction": 0.0,
            "max_saturated_fraction": 0.0,
            "n_nodes": self._regularisation_stats.get("n_nodes", 0) or 0,
        }
        self.material_properties.eccentricity0 = np.array([0.0, 0.0, ecc], dtype=float)

        print(
            f"Max initial pressure: {self.p.vector().max()*self.material_properties.p0/1e6} MPa"
        )
        print(
            f"Max initial deformation: {self.delta.vector().max()*self.material_properties.c} mm"
        )

        # Tolerances and loop controls (pressure-only stop criterion)
        pressure_relaxation_tol = float(self.inner_tol_p)
        eta_relax_trigger = (
            float(self.eta_relax_trigger_factor) * pressure_relaxation_tol
        )

        if eta_relax_trigger <= 0.0:
            eta_relax_trigger = pressure_relaxation_tol

        max_relaxation_iterations = 300
        relaxation_iteration = 0
        pressure_residual = np.inf
        # Pressure relaxation factor (can be overridden by setting self.theta_p_relax)
        theta_p = float(getattr(self, "theta_p_relax", 0.5))

        def _rebuild_deformation():
            # Use *current* p and the given (fixed) dP to rebuild δ
            delta = self.calculate_deformation(self.p, self.K)
            if hasattr(self, "dP"):
                delta += self.calculate_deformation(self.dP, self.K)
            self.delta.vector()[:] = delta

        print("Starting EHL deformation loop (no dP/dQ relaxation; α,β fixed).")
        with self._suspend_balance_cache():
            while ((pressure_residual > pressure_relaxation_tol)) and (
                relaxation_iteration < max_relaxation_iterations
            ):
                # Update h with current δ (from p and fixed dP)
                _rebuild_deformation()
                print(f"Current regularisation parameter {self.h_regularisation_param}")
                print(f"Current alpha value: {float(self.alpha)}")
                print(f"Current beta value: {float(self.beta)}")
                self.update_contact_separation(
                    self.material_properties.eccentricity0,
                    HMMState=HMMState,
                    transientState=transientState,
                    EHLState=True,
                )
                # One Newton step on p (no alpha-continuation; beta frozen)
                p_prev = self.p.vector().get_local().copy()
                _xi, _ = self.soderfjall_solve(
                    HMMState=HMMState,
                    transientState=transientState,
                    postprocess=False,
                    full=False,
                )
                print(
                    f"Max pressure after Newton: {self.p.vector().max()*self.material_properties.p0/1e6} MPa"
                )
                print(
                    f"Min film thickness after Newton: {self.h.vector().min()*self.material_properties.c/1e-6} um"
                )

                # --- Pressure relaxation (mixing) ---
                p_trial = self.p.vector().get_local()  # result from Newton step
                # p_mixed = (1-θ) p_prev + θ p_trial
                p_mixed = (1.0 - theta_p) * p_prev + theta_p * p_trial
                self.p.vector()[:] = p_mixed
                self.p.vector().apply("insert")

                # Rebuild δ and h *for the mixed pressure* and compute residual w.r.t previous state
                _rebuild_deformation()
                self.update_contact_separation(
                    self.material_properties.eccentricity0,
                    HMMState=HMMState,
                    transientState=transientState,
                    EHLState=True,
                )
                pressure_residual = self.relative_residual(p_mixed, p_prev)
                self._update_eta_relaxation(
                    pressure_residual, relaxation_iteration, eta_relax_trigger
                )
                print(
                    f"Relaxation iter {relaxation_iteration}, "
                    f"p-res {pressure_residual:.3e}, p-mix {theta_p:.3f}"
                )
                # Export the mixed pressure field for this iteration
                self.export_series(field="p", t=relaxation_iteration)

                relaxation_iteration += 1

        # Post-loop: gradients, forces, logging, residual guard
        try:
            self.calc_gradP()
        except Exception:
            pass

        self.calc_force()
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
            f"Offset: [{self.material_properties.eccentricity0[2] / self.material_properties.length_ratio: >3.8%}], "
            f'Load: [{" ".join(f"{x: >8.5f}" for x in load)}], '
            f'Force: [{" ".join(f"{x: >8.5f}" for x in self.force)}], '
            f"Difference: [{difference[2]: >3.2e}], "
            f"difference norm: {np.linalg.norm(difference[2]): >9.4e}"
        )
        print(
            f"Maximum pressure MPa: {self.p.vector().max()*self.material_properties.p0/1e6}"
        )
        print(
            f"Minimum film thickness um: {self.h.vector().min()*self.material_properties.c/1e-6}"
        )

        denom = abs(load[2]) if abs(load[2]) > 0 else 1.0
        returned_residual_nd = float(difference[2]) / denom
        guarded_residual_nd = self._apply_regularisation_guard(returned_residual_nd)
        try:
            frac_pre_end = 100.0 * float(
                self._regularisation_stats.get("last_fraction", 0.0)
            )
            frac_sat_end = 100.0 * float(
                self._regularisation_stats.get("last_saturated_fraction", 0.0)
            )
            print(
                f"[REG-GUARD] end-state: pre_end={frac_pre_end:.2f}%, "
                f"sat_end={frac_sat_end:.2f}% → returned ND residual: {guarded_residual_nd:.3e}"
            )
        except Exception:
            pass
        if not self._cache_suppressed:
            self._balance_cache[key] = (
                _xi if "_xi" in locals() else None,
                guarded_residual_nd,
            )
        print(f"[RESID] dim={float(difference[2]):+.3e}, nd={guarded_residual_nd:+.3e}")
        return (_xi if "_xi" in locals() else None), guarded_residual_nd

    def EHL_balance_equation(self, eccentricity, HMMState=False, transientState=False):
        """
        Load-balance residual (normalised z-component) with *pressure-only* relaxation.
        Deformation is recomputed from the relaxed pressure without any δ mixing/capping.
        """
        ecc = self._as_scalar(eccentricity)
        key = self._cache_key(ecc, HMMState, transientState)

        if not self._cache_suppressed and key in self._balance_cache:
            self._memo_count += 1
            cached_xi, cached_resid = self._balance_cache[key]
            # if not isinstance(cached, (tuple, list)) or len(cached) != 2:
            #     return (None, float(cached))
            return cached_xi, cached_resid


        print(f"Solving with eccentricity {eccentricity}")
        # (Re)allow beta updates for this outer call
        if transientState:
            # Keep beta frozen to the loaded value; otherwise rhoprev becomes inconsistent.
            self._beta_freeze_active = True
            self._beta_freeze_just_set = False
        else:
            self._beta_freeze_active = False
            self._beta_freeze_just_set = False
        # reset smoothing decay state for this outer evaluation
        self._reset_reg_decay()
        self._reset_eta_relaxation()
        # ---- reset regularisation stats for this outer evaluation ----
        self._regularisation_stats = {
            "ever_active": False,
            "last_count": 0,
            "last_fraction": 0.0,
            "max_fraction": 0.0,
            "last_saturated_fraction": 0.0,
            "max_saturated_fraction": 0.0,
            "n_nodes": self._regularisation_stats.get("n_nodes", 0) or 0,
        }

        # Basic state setup
        self.material_properties.eccentricity0 = np.array([0.0, 0.0, ecc], dtype=float)

        print(
            f"Max initial pressure: {self.p.vector().max()*self.material_properties.p0/1e6} MPa"
        )
        print(
            f"Max initial deformation: {self.delta.vector().max()*self.material_properties.c} mm"
        )
        print(f"Current regularisation parameter {self.h_regularisation_param}")
        print(f"Current alpha value: {float(self.alpha)}")
        print(f"Current beta value: {float(self.beta)}")
        # First uncavitated-magnitude estimate + one Newton to seed p
        if self._skip_function_space_setup == False:
            self.setup_function_spaces()
            self.initialise_bc()
            self.initialise_velocity()
            self.update_contact_separation(
                self.material_properties.eccentricity0,
                HMMState=HMMState,
                transientState=transientState,
                EHLState=False,   # no deformation yet for beta-estimate seed
            )
            print(
                f"Min Film thickness: {self.h.vector().min()*self.material_properties.c:.4e}"
            )

            self.est_pmag_problem(HMMState=HMMState, transientState=transientState)
            print(f"Running the solve for estimating beta")
            self.fenicssolve()


        # ---------------------------
        # BASIC RELAXATION PARAMETERS
        # ---------------------------
        pressure_relaxation_tol = float(self.inner_tol_p)
        eta_relax_trigger = (
            float(self.eta_relax_trigger_factor) * pressure_relaxation_tol
        )
        if eta_relax_trigger <= 0.0:
            eta_relax_trigger = pressure_relaxation_tol
        deformation_relaxation_tol = float(self.inner_tol_d)

        print(
            f"Using inner tols: p={pressure_relaxation_tol:.2e}, δ={deformation_relaxation_tol:.2e} "
            f"(min iters={self.inner_min_iters})"
        )

        # Pressure under-relaxation (fixed)
        theta_p = 0.2  # conservative pressure under-relaxation

        # --- Symmetric ramp-up controls (pressure mixing) ---
        # If the pressure residual keeps decreasing for a few consecutive steps,
        # gently increase theta_p up to a safe cap to shorten the "long tail".
        theta_p_min = 0.05
        theta_p_cap = 0.5
        theta_p_growth = 1.25  # multiplicative step when ramping up
        dec_needed = 3  # consecutive decreases required
        dec_count = 0  # counter of consecutive decreases

        # ONLY-P RELAX: remove δ ramp & caps entirely

        max_relaxation_iterations = 300
        relaxation_iteration = 0
        pressure_residual = 1.0
        deformation_residual = 1.0
        prev_p_res = np.inf
        prev_d_res = np.inf

        print("Starting EHL deformation loop (pressure-only relaxation)...")
        print(
            f"Initial residuals: p={pressure_residual:.2e}, δ={deformation_residual:.2e}"
        )
        # --- Require the regulariser to be fully removed before exiting ---
        # We consider regularisation "active" if either the smoothing S>0
        # or if the end-of-iteration fractions report that any nodes are still
        # being regularised. Add a small hysteresis (need a couple of clean
        # iterations) to avoid stopping on a noisy boundary.
        reg_clear_needed = int(getattr(self.solver_params, "reg_clear_needed", 2))
        reg_frac_tol = float(
            getattr(self.solver_params, "reg_clear_fraction_tol", 1e-6)
        )
        reg_clear_streak = 0

        def _regularisation_active() -> bool:
            stats = getattr(self, "_regularisation_stats", {}) or {}
            frac_pre = float(stats.get("last_fraction", 0.0) or 0.0)
            frac_sat = float(stats.get("last_saturated_fraction", 0.0) or 0.0)
            frac_end = max(frac_pre, frac_sat)
            # active if smoothing has not yet reached zero OR if any nontrivial fraction is still flagged
            return (getattr(self, "_reg_decay_S_current", 0.0) > 0.0) or (
                frac_end > reg_frac_tol
            )

        def _need_to_continue() -> bool:
            return reg_clear_streak < reg_clear_needed

        with self._suspend_balance_cache():
            while (
                (pressure_residual > pressure_relaxation_tol)
                or (deformation_residual > deformation_relaxation_tol)
                or (relaxation_iteration < self.inner_min_iters)
                or _regularisation_active()  # <-- do not exit while any regularisation remains
                or _need_to_continue()  # <-- require a couple of clean iterations after S→0
            ) and relaxation_iteration < max_relaxation_iterations:
                # print(
                # f"Relaxation iteration {relaxation_iteration}, "
                # f"p-res {pressure_residual:.3e}, δ-res {deformation_residual:.3e}"
                # )

                # Current states
                p_old = self.p.vector().get_local().copy()
                delta_old = self.delta.vector().get_local().copy()

                # Update h from current δ before solving p
                self.update_contact_separation(
                    self.material_properties.eccentricity0,
                    HMMState=HMMState,
                    transientState=transientState,
                    EHLState=True,
                )
                # print(
                # f"Min Film thickness mm: {self.h.vector().min()*self.material_properties.c:.4e}"
                # )

                # One Newton: full path only on the very first inner iter
                xi, _ = self.soderfjall_solve(
                    HMMState=HMMState,
                    transientState=transientState,
                    postprocess=True,
                    full=(relaxation_iteration == 0 and (not transientState)),
                )

                # Freeze beta once we're 'close' to convergence to avoid jitter
                try:
                    _freeze_tol = getattr(self.solver_params, "beta_freeze_p_tol", 5e-4)
                    _freeze_after = getattr(
                        self.solver_params, "beta_freeze_min_iters", 20
                    )
                except AttributeError:
                    _freeze_tol, _freeze_after = 5e-4, 20

                if (
                    (not getattr(self, "_beta_freeze_active", False))
                    and (relaxation_iteration >= _freeze_after)
                    and (prev_p_res < _freeze_tol)
                ):
                    self._beta_freeze_active = True
                    self._beta_freeze_just_set = True
                    # print(
                    #     f"[BETA] Freezing beta at {float(self.beta):.6g} "
                    #     f"(p-res={prev_p_res:.3e}, iter={relaxation_iteration})"
                    # )

                # Export (as before)
                # self.export_series(field="h", t=relaxation_iteration)
                # self.export_series(field="p", t=relaxation_iteration)

                # --- PRESSURE MIXING (unchanged) ---
                p_trial = self.p.vector().get_local().copy()
                p_new = (1.0 - theta_p) * p_old + theta_p * p_trial
                # Optional clamp if desired:
                # p_new = np.maximum(p_new, 0.0)

                self.p.vector().set_local(p_new)
                self.p.vector().apply("insert")

                # --- ONLY-P RELAX: recompute δ from the *mixed* pressure, no δ mixing ---
                delta_new = self.calculate_deformation(self.p, self.K)

                # Write back δ fields directly
                self.delta.vector()[:] = delta_new

                # Rebuild h with new δ (for residuals & next iter)
                self.update_contact_separation(
                    self.material_properties.eccentricity0,
                    HMMState=HMMState,
                    transientState=transientState,
                    EHLState=True,
                )

                # Residuals
                pressure_residual = self.relative_residual(p_new, p_old)
                deformation_residual = self.relative_residual(delta_new, delta_old)
                self._update_eta_relaxation(
                    pressure_residual, relaxation_iteration, eta_relax_trigger
                )

                # --- Trigger & drive smoothing S decay once 'close enough' ---
                try:
                    trigger = float(self.reg_decay_trigger_factor) * float(
                        pressure_relaxation_tol
                    )
                except Exception:
                    trigger = 2.0 * float(pressure_relaxation_tol)
                if (
                    (not self._reg_decay_active)
                    and self._eta_relaxation_complete()
                    and (pressure_residual <= trigger)
                ):
                    self._reg_decay_active = True
                    self._reg_decay_start_iter = relaxation_iteration
                    self._reg_decay_S0 = float(self._reg_decay_S_current)
                    # print(
                    #     f"[REG-S] Starting smoothing decay: S0={self._reg_decay_S0:.3e}, "
                    #     f"trigger={trigger:.3e}, iter={relaxation_iteration}"
                    # )
                if self._reg_decay_active:
                    k = max(int(self.reg_decay_iters), 1)
                    prog = max(
                        0.0,
                        min(
                            1.0,
                            (relaxation_iteration - self._reg_decay_start_iter + 1) / k,
                        ),
                    )
                    self._reg_decay_S_current = (1.0 - prog) * self._reg_decay_S0
                    # Nudge strictly to zero at the end to fully remove the floor
                    if prog >= 1.0:
                        self._reg_decay_S_current = 0.0
                    # print(
                    #     f"[REG-S] iter={relaxation_iteration} "
                    #     f"S={self._reg_decay_S_current:.3e} (prog={prog:.2f})"
                    # )

                # Back-off or ramp-up for pressure (ONLY-P RELAX)
                if pressure_residual > 1.2 * prev_p_res:
                    # residual worsened noticeably → back off and reset ramp counter
                    theta_p = max(0.5 * theta_p, theta_p_min)
                    dec_count = 0
                else:
                    # Track consecutive decreases; ignore tiny noise
                    # Treat as a "decrease" if down by at least ~1% (robust to jitter)
                    if prev_p_res < np.inf and pressure_residual <= 0.99 * prev_p_res:
                        dec_count += 1
                    else:
                        dec_count = 0

                    # When things are smooth, gently increase theta_p (up to cap)
                    if dec_count >= dec_needed:
                        new_theta = min(theta_p * theta_p_growth, theta_p_cap)
                        # Only log if we actually changed it
                        if new_theta > theta_p:
                            # (optional) you can print a one-liner here if you want visibility
                            # print(f"[P-MIX] Ramp-up: {theta_p:.3f} → {new_theta:.3f}")
                            theta_p = new_theta
                        dec_count = 0

                prev_p_res = pressure_residual
                prev_d_res = deformation_residual

                # print(
                #     f"Relaxation iterat {relaxation_iteration}, "
                #     f"p-res {pressure_residual:.3e}, δ-res {deformation_residual:.3e}, "
                #     f"p-mix {theta_p:.3f} (δ has no mixing)"
                # )

                relaxation_iteration += 1
                # --- Track whether regularisation is fully cleared ---
                # If nothing is active, count a "clean" iteration; otherwise reset the streak.
                try:
                    if not _regularisation_active():
                        reg_clear_streak += 1
                    else:
                        reg_clear_streak = 0
                except Exception:
                    # be conservative if anything goes wrong
                    reg_clear_streak = 0
        print(
            f"Total relaxation iterations: {relaxation_iteration}, "
            f"final p-res: {pressure_residual:.3e}, final δ-res: {deformation_residual:.3e}"
        )

        # -------------------------
        # FINAL "POLISH" NEWTON STEP
        # -------------------------
        # Purpose: certify the final problem without any pressure mixing,
        #          *without* re-regularising h or changing α,β,S.
        # Strategy: do ONE Newton step at the current (p, δ, h) and keep it
        #           only if it strictly improves the pressure change norm.
        try:
            p_before = self.p.vector().get_local().copy()
            # # Temporarily freeze beta (avoid small α/β jitter during polish)
            """
            # _beta_freeze_backup = bool(getattr(self, "_beta_freeze_active", False))
            # self._beta_freeze_active = True
            # self.delta.vector()[:] = self.calculate_deformation(self.p, self.K)
            # self.alpha.assign(float(0.001))  # keep current alpha
            """
            # Temporarily freeze beta (avoid small α/β jitter during polish)
            _beta_freeze_backup = bool(getattr(self, "_beta_freeze_active", False))
            _alpha_backup = float(getattr(self, "alpha", 0.0))
            self._beta_freeze_active = True
            self.delta.vector()[:] = self.calculate_deformation(self.p, self.K)
            self.alpha.assign(_alpha_backup)  # keep current alpha
            self.update_contact_separation(
                self.material_properties.eccentricity0,
                HMMState=HMMState,
                transientState=transientState,
                EHLState=True,
            )
            # Do NOT call update_contact_separation() here—keep the same h.
            # One Newton step only (full=False), no postprocess.
            _xi_polish, _ = self.soderfjall_solve(
                HMMState=HMMState,
                transientState=transientState,
                postprocess=True,
                full=False,
            )
            p_after = self.p.vector().get_local().copy()
            polish_res = self.relative_residual(p_after, p_before)
            print(f"[POLISH] Attempted final Newton step, Δp_rel={polish_res}")
            # Accept only if it improves the change by a meaningful margin.
            # (Use ~0.7× inner tol to avoid noise-triggered flips.)
            if not np.isfinite(polish_res) or (
                polish_res > max(0.7 * pressure_relaxation_tol, pressure_residual)
            ):
                # Revert to mixed solution if not clearly beneficial.
                self.p.vector().set_local(p_before)
                self.p.vector().apply("insert")
                # restore δ consistent with reverted p
                _d = self.calculate_deformation(self.p, self.K)
                self.delta.vector()[:] = _d
                print(
                    "[POLISH] Step rejected (no improvement). Kept mixed-solution state."
                )
            else:
                # Keep polished pressure; update δ to match kept pressure
                _d = self.calculate_deformation(self.p, self.K)
                self.delta.vector()[:] = _d
                print(f"[POLISH] Step accepted (Δp_rel={polish_res:.3e}).")
        except Exception as _polish_err:
            # Be conservative—do nothing if polish fails for any reason.
            print(
                f"[POLISH] Skipped due to error: {type(_polish_err).__name__}: {_polish_err}"
            )
        finally:
            # Restore beta freeze state
            """
            # try:
            #     self._beta_freeze_active = _beta_freeze_backup
            # except Exception:
            #     pass
            """
            try:
                self._beta_freeze_active = _beta_freeze_backup
            except Exception:
                pass
            try:
                self.alpha.assign(_alpha_backup)
            except Exception:
                pass

        # Outer residual and logging (unchanged)
        self.calc_force()
        load = (
            self.current_load
            * np.array(self.material_properties.load_orientation)
            * 2
            * np.pi
            / 3
        )
        difference = load + self.force
        try:
            self._last_outer_resid = abs(difference[2]) / (
                abs(load[2]) if abs(load[2]) > 0 else 1.0
            )
        except Exception:
            self._last_outer_resid = np.nan
        self.main_residuals.append(np.linalg.norm(difference[2]))

        print(
            f"Offset: [{self.material_properties.eccentricity0[2] / self.material_properties.length_ratio: >3.8%}], "
            f'Load: [{" ".join(f"{x: >8.5f}" for x in load)}], '
            f'Force: [{" ".join(f"{x: >8.5f}" for x in self.force)}], '
            f"Difference: [{difference[2]: >3.2e}], "
            f"difference norm: {np.linalg.norm(difference[2]): >9.4e}"
        )
        print(
            f"Maximum pressure MPa: {self.p.vector().max()*self.material_properties.p0/1e6}"
        )
        print(
            f"Minimum film thickness um: {self.h.vector().min()*self.material_properties.c/1e-6}"
        )
        print(
            f"Maximum deformation um: {self.delta.vector().max()*self.material_properties.c/1e-6}"
        )

        # ---- normalise residual; apply regularisation (end-state only) penalty ----
        denom = abs(load[2]) if abs(load[2]) > 0 else 1.0
        returned_residual_nd = float(difference[2]) / denom

        guarded_residual_nd = self._apply_regularisation_guard(returned_residual_nd)
        try:
            # Log with end-of-loop fractions for clarity
            frac_pre_end = 100.0 * float(
                self._regularisation_stats.get("last_fraction", 0.0)
            )
            frac_sat_end = 100.0 * float(
                self._regularisation_stats.get("last_saturated_fraction", 0.0)
            )
            msg = (
                f"[REG-GUARD] end-state: pre_end={frac_pre_end:.2f}%, "
                f"sat_end={frac_sat_end:.2f}% → returned ND residual: {guarded_residual_nd:.3e}"
            )

            print(msg)
        except Exception:
            pass

        if not self._cache_suppressed:
            self._balance_cache[key] = (xi, guarded_residual_nd)
        print(f"[RESID] dim={float(difference[2]):+.3e}, nd={guarded_residual_nd:+.3e}")
        return xi, guarded_residual_nd

    def solve_loadbalance_EHL(self, HMMState=False, transientState=False):
        """
        Use scipy.optimize.root to find the eccentricity that balances the equation.
        """
        initial_guess = self.material_properties.eccentricity0[2]
        self.main_residuals = []
        tol = self.solver_params.load_balance_rtol
        # Ensure stale cached residuals from previous solves do not short-circuit
        # the optimisation; every evaluation during this solve should recompute
        # the macroscale state for the requested eccentricity.
        self._balance_cache.clear()

        def res_only_balance(e, HMMStatein, transientStatein):
            _, resid = self.EHL_balance_equation(e, HMMState=HMMStatein, transientState=transientStatein)
            return resid
        
        result = root(
            res_only_balance,
            initial_guess,
            args=(HMMState, transientState),
            method="hybr",
            tol=tol,
        )

        if result.success:
            eccentricity_new = self._as_scalar(result.x)
            self.material_properties.eccentricity0 = np.array(
                [0.0, 0.0, eccentricity_new], dtype=float
            )
            print(
                f"Optimization successful: {result.success}, "
                f'Offset: [{" ".join(f"{x / self.material_properties.length_ratio: >7.2%}" for x in self.material_properties.eccentricity0)}]'
            )

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
            h = h - self.local_deformation()
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

    def project_fluidfrac(self):
        self.FluidFrac = project(
            self.FluidFraction,
            self.V,
            solver_type="cg",
            preconditioner_type="hypre_amg",
        )

        self.gradF = project(grad(self.FluidFrac), self.W)

    def solve_loadbalance_secant(self, HMMState=False, transientState=False):
        """
        Use a secant-style iteration to find the eccentricity that balances the equation.
        """
        initial_guess = self.material_properties.eccentricity0[2]
        self.main_residuals = []
        tol = self.solver_params.load_balance_rtol
        max_iter = getattr(self.solver_params, "load_balance_max_iterations", 50)
        max_ecc = 1.05 * self.material_properties.length_ratio

        def _clamp(value: float) -> float:
            return min(max(value, 0.0), max_ecc)

        # Ensure stale cached residuals from previous solves do not short-circuit
        # the optimisation; every evaluation during this solve should recompute
        # the macroscale state for the requested eccentricity.
        self._balance_cache.clear()

        x0 = _clamp(self._as_scalar(initial_guess))
        print(f"x0 for secant: {x0}")
        _, f0 = self.EHL_balance_equation(x0, HMMState, transientState)
        if abs(f0) <= tol:
            print(f"[SECANT] Initial guess is already converged with residual {f0:.3e}")
            eccentricity_new = x0
        else:
            # Sign-aware seed based on physics: f<0 ⇒ over-supported ⇒ reduce e; f>0 ⇒ increase e
            adj_abs = max(
                1e-6 * self.material_properties.length_ratio, 1e-10
            )  # absolute floor
            adj_rel = 5e-7  # ~0.05% relative step (was 0.0005 multiplier)
            adj = max(adj_abs, adj_rel * max(x0, 1.0))
            print(f"Adj for secant seed: {adj}")

            x1 = _clamp(x0 - adj if f0 < 0.0 else x0 + adj)
            if x1 == x0:
                # If clamped at a boundary, flip once to ensure a nonzero move
                x1 = _clamp(x0 + adj if f0 < 0.0 else x0 - adj)

            _, f1 = self.EHL_balance_equation(x1, HMMState, transientState)
            # If the first move made |f| worse, shrink once and retry in the SAME sign direction
            if abs(f1) >= abs(f0):
                adj *= 0.25
                x1 = _clamp(x0 - adj if f0 < 0.0 else x0 + adj)
                if x1 == x0:
                    x1 = _clamp(x0 + adj if f0 < 0.0 else x0 - adj)
                _, f1 = self.EHL_balance_equation(x1, HMMState, transientState)
            eccentricity_new = None

            for iteration in range(max_iter):
                if abs(f1) <= tol:
                    eccentricity_new = x1
                    print(
                        f"[SECANT] Converged in {iteration + 1} iterations with residual {f1:.3e}"
                    )
                    break

                denom = f1 - f0
                if abs(denom) < 1e-14:
                    step = -np.sign(f1) * adj if f1 != 0 else adj
                    x2 = _clamp(x1 + step)
                else:
                    x2 = _clamp(x1 - f1 * (x1 - x0) / denom)

                if x2 == x1:
                    step = -np.sign(f1) * adj if f1 != 0 else adj
                    x2 = _clamp(x2 + step)

                # One-shot backtrack safeguard: ensure progress in |f|
                _, f_trial = self.EHL_balance_equation(x2, HMMState, transientState)
                if abs(f_trial) > abs(f1):
                    x2 = _clamp(0.5 * (x2 + x1))
                    _, f_trial = self.EHL_balance_equation(x2, HMMState, transientState)

                x0, f0 = x1, f1
                x1, f1 = x2, f_trial

            if eccentricity_new is None:
                if abs(f1) <= tol:
                    eccentricity_new = x1
                else:
                    print(
                        "Load balance secant solver failed to converge within "
                        f"{max_iter} iterations (residual {f1})."
                    )
                    return None

        self.material_properties.eccentricity0 = np.array(
            [0.0, 0.0, eccentricity_new], dtype=float
        )
        # backup the deformation solution
        backupdeformation = self.delta.vector()[:]
        # recalculate deformation at final eccentricity and pressure
        newdeformation = self.calculate_deformation(self.p, self.K)
        print(
            f"Change in deformation norm after final calc: {np.linalg.norm(newdeformation - backupdeformation):.3e}"
        )

        print(
            f"Optimization successful: True, "
            f'Offset: [{" ".join(f"{x / self.material_properties.length_ratio: >7.2%}" for x in self.material_properties.eccentricity0)}]'
        )

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
        h = h - self.local_deformation()
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

    def construct_transient_xi(self, xi_new, xi_old):
        h_prev, h_new = xi_old[0, :], xi_new[0, :]
        p_prev, p_new = xi_old[1, :], xi_new[1, :]

        dhdt = self.calc_temporal_gradient(h_new, h_prev)
        dpdt = self.calc_temporal_gradient(p_new, p_prev)
        dhdt = dhdt[np.newaxis, :]
        dpdt = dpdt[np.newaxis, :]
        base = xi_old[:12, :]
        xi_out = np.concatenate((base, dhdt, dpdt), axis=0)
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
        gradpx = xi[5] * self.material_properties.p0 / self.material_properties.Rc
        gradpy = xi[6] * self.material_properties.p0 / self.material_properties.Rc
        gradpz = xi[7] * self.material_properties.p0 / self.material_properties.Rc
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

    def rotate_Q(self):
        R = self.R
        Q_rot = rotate_array(self.Q, R)
        Q_r = np.array(Q_rot).reshape((-1, 3))
        return Q_r

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
            self._roelands_effective(self.p * self.material_properties.p0)
            / self.material_properties.eta0
        )
        Q_exp = self.U * self.h - self.h**3 * self.gradP / (
            (self.FluidFraction + 0.001) / (1 + 0.001) * roelands_eta
        )  ###################Is this right still?
        self.Q = project(Q_exp, self.W)

    def local_deformation(self):
        # returns values as an array
        if not hasattr(self, "dP"):
            return (
                self.material_properties.p0 / self.material_properties.k_spring
            ) * self.p.vector()[:]
        else:
            return (self.material_properties.p0 / self.material_properties.k_spring) * (
                self.p.vector()[:] + self.dP.vector()[:]
            )

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

    # --------- DIAGNOSTICS: snapshot & compare (init vs reloaded) ----------
    def snapshot_state(self):
        """
        Capture a reference snapshot of the *current* in-memory solver state
        (arrays and key scalars) for later comparison.
        Returns a plain dict safe to store/print.
        """
        ref = {}
        # Arrays (copy to decouple from FEniCS storage)
        try:
            ref["pressure"] = self.p.vector().get_local().copy()
        except Exception:
            ref["pressure"] = None
        try:
            ref["deformation"] = self.delta.vector().get_local().copy()
        except Exception:
            ref["deformation"] = None
        try:
            ref["film_thickness"] = self.h.vector().get_local().copy()
        except Exception:
            ref["film_thickness"] = None
        # Scalars
        try:
            ref["alpha"] = float(self.alpha)
        except Exception:
            ref["alpha"] = None
        try:
            ref["beta"] = float(self.beta)
        except Exception:
            ref["beta"] = None
        try:
            # store the eccentricity (z component in your setup)
            ref["eccentricity_z"] = float(self.material_properties.eccentricity0[2])
        except Exception:
            ref["eccentricity_z"] = None
        # Regularisation/scalars that may affect rebuilds
        ref["h_regularisation_param"] = float(
            getattr(self, "h_regularisation_param", 0.0)
        )
        ref["h_min"] = float(getattr(self, "h_min", 0.0))
        return ref

    def compare_loaded_state(self, reference):
        """
        Compare the *current* solver state (after you have reloaded arrays with
        `load_state(...)`) to a previously captured `reference` snapshot.
        Prints max absolute and relative differences for key fields & scalars.
        """
        import numpy as _np

        def _maxabs(a, b):
            if a is None or b is None:
                return _np.nan
            return float(_np.max(_np.abs(a - b))) if a.size and b.size else 0.0

        def _maxrel(a, b):
            if a is None or b is None:
                return _np.nan
            denom = _np.maximum(_np.abs(a), _np.abs(b))
            return (
                float(_np.max(_np.abs(a - b) / (denom + 1e-16)))
                if a.size and b.size
                else 0.0
            )

        print("\n[DIAGNOSTIC] === INIT vs RELOADED (current state) ===")

        # Scalars
        cur_alpha = float(getattr(self, "alpha", 0.0))
        cur_beta = float(getattr(self, "beta", 0.0))
        cur_eccz = float(
            getattr(self.material_properties, "eccentricity0", [0, 0, 0])[2]
        )
        cur_S = float(getattr(self, "h_regularisation_param", 0.0))
        cur_hmin = float(getattr(self, "h_min", 0.0))

        def _print_scalar(name, cur, ref):
            if ref is None:
                print(f"{name:25s}: (no reference)")
                return
            dabs = abs(cur - ref)
            drel = dabs / (abs(ref) + 1e-16)
            print(
                f"{name:25s}: Δabs={dabs:.3e}, Δrel={drel:.3e}  (cur={cur:.6g}, ref={ref:.6g})"
            )

        _print_scalar("alpha", cur_alpha, reference.get("alpha"))
        _print_scalar("beta", cur_beta, reference.get("beta"))
        _print_scalar("eccentricity_z", cur_eccz, reference.get("eccentricity_z"))
        _print_scalar(
            "h_regularisation_param", cur_S, reference.get("h_regularisation_param")
        )
        _print_scalar("h_min", cur_hmin, reference.get("h_min"))

        # Arrays
        try:
            p_cur = self.p.vector().get_local()
        except Exception:
            p_cur = None
        try:
            d_cur = self.delta.vector().get_local()
        except Exception:
            d_cur = None
        try:
            h_cur = self.h.vector().get_local()
        except Exception:
            h_cur = None

        p_ref = reference.get("pressure")
        d_ref = reference.get("deformation")
        h_ref = reference.get("film_thickness")

        print(
            "\n[pressure]     Δabs_max = %.3e, Δrel_max = %.3e"
            % (_maxabs(p_cur, p_ref), _maxrel(p_cur, p_ref))
        )
        print(
            "[deformation]  Δabs_max = %.3e, Δrel_max = %.3e"
            % (_maxabs(d_cur, d_ref), _maxrel(d_cur, d_ref))
        )
        print(
            "[film_thickn.] Δabs_max = %.3e, Δrel_max = %.3e"
            % (_maxabs(h_cur, h_ref), _maxrel(h_cur, h_ref))
        )
        print("[DIAGNOSTIC] ==================================================\n")

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
