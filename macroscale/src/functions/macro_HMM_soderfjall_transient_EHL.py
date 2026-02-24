import os
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

        marker = MeshFunction('size_t', meshC, meshC.topology().dim() - 1, 0)
        cylinder_subdomain = CylinderSubdomain2(self.mesh_parameters)  # Use cylindersubdomain3 for the point pressure
        complement_cylinder_subdomain = ComplementCylinderSubdomain(self.mesh_parameters)
        cut  = CutPlane(self.mesh_parameters)
        complement_cylinder_subdomain.mark(marker, 2)
        cylinder_subdomain.mark(marker, 1)
        cut.mark(marker, 3)

        submesh = MeshView.create(marker, 1)


        self.marker = marker
        self.meshC = meshC
        self.meshF = submesh

class CylinderSubdomain2(SubDomain):
    def __init__(self, mp):
        self.mp = mp
        super().__init__()

    def inside(self, x, on_boundary):
        if not on_boundary:
            return False

        # Unapply displacement & scaling to get normalized coords of the original unit sphere
        X = (x[0] - self.mp.CupDisplace[0]) / self.mp.CupScale[0]
        Y = (x[1] - self.mp.CupDisplace[1]) / self.mp.CupScale[1]
        Z = (x[2] - self.mp.CupDisplace[2]) / self.mp.CupScale[2]

        # Cylindrical footprint in x–y (same as before, but on normalized coords)
        cyl_ok = (X*X + Y*Y) <= (1.0 + self.mp.tol)
        if not cyl_ok:
            return False

        # z-range gate (use your existing bounds; adjust as needed)
        z_ok = (-1.0 <= Z <= (1.0 + self.mp.tol))   # or your previous z0..z1 mapping
        if not z_ok:
            return False

        # Keep all points away from the cut plane…
        ytol = self.mp.tol
        if abs(Y) > ytol:
            return True

        # …but ON the cut plane (|Y|≈0), only keep those on the spherical surface (rim)
        r2 = X*X + Y*Y + Z*Z
        sph_ok = abs(r2 - 1.0) <= (5 * self.mp.tol)  # slightly looser tol for radius test
        return sph_ok

class CylinderSubdomain3(SubDomain):
    def __init__(self, mesh_parameters):
        self.mesh_parameters = mesh_parameters
        super().__init__()

    def inside(self, x, on_boundary):
        # Check if a point is within the specified cylindrical area and on the boundary
        z0 = -1
        return on_boundary and (
            (x[0] ** 2) / self.mesh_parameters.CupScale[0] ** 2 + (x[1] ** 2) / self.mesh_parameters.CupScale[1] ** 2
            <= (0.1 + self.mesh_parameters.tol)
        ) and (z0 <= x[2] <= (self.mesh_parameters.CupScale[2] + self.mesh_parameters.tol))

class CutPlane(SubDomain):
    def __init__(self, mp):
        self.mp = mp
        super().__init__()
    def inside(self, x, on_boundary):
        if not on_boundary:
            return False
        # normalize coords
        X = (x[0]-self.mp.CupDisplace[0]) / self.mp.CupScale[0]
        Y = (x[1]-self.mp.CupDisplace[1]) / self.mp.CupScale[1]
        Z = (x[2]-self.mp.CupDisplace[2]) / self.mp.CupScale[2]
        return abs(Y) <= 5*self.mp.tol    # “on” the cut plane (y≈0)
    

class ComplementCylinderSubdomain(SubDomain):
    def __init__(self, mesh_parameters):
        self.mesh_parameters = mesh_parameters
        super().__init__()

    def inside(self, x, on_boundary):
        z0=-1
        # Return True for points not within the specified cylindrical area but on the boundary
        return (on_boundary and not (
            (x[0] ** 2) / self.mesh_parameters.CupScale[0] ** 2 + (x[1] ** 2) / self.mesh_parameters.CupScale[1] ** 2
            <= (1 - self.mesh_parameters.tol))) or (on_boundary and (
            (x[0] ** 2) / self.mesh_parameters.CupScale[0] ** 2 + (x[1] ** 2) / self.mesh_parameters.CupScale[1] ** 2
            <= (1 - self.mesh_parameters.tol)) and (x[2] > self.mesh_parameters.CupScale[2] + self.mesh_parameters.tol))


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
    return eta00 #exp(ln_eta)


class EHLSolver:
    def __init__(self, meshin, material_properties, solver_params, output_folder):
        """
        Initialize the HydroSolver with mesh data, material properties, and solver parameters.
        """
        self.output_folder = output_folder
        self.meshin = meshin
        self.cav = "on"
        self.beta = 1
        self.h_regularisation_param = 1e-2 #5e-3
        self.alpha = Constant(1e4)
        self.material_properties = material_properties
        self.solver_params = solver_params
        self.t = 0.0
        self.t_prev = 0.0
        self.Tend = self.solver_params.Tend
        self.dt = self.solver_params.dt
        self.transient_flag = False
        self.load_influence_matrix()
        self.split_influence_matrix()
        self.setup_function_spaces()
        self.initialise_bc()
        self.initialise_velocity()
        self.update_contact_separation(
            self.material_properties.eccentricity0, HMMState=False, transientState=False, EHLState=False
        )
        self.main_residuals = []
        self.init_hprev()
        self.init_rhoprev()
        self.calc_current_load(self.t)
        self.h_file = File("data/output/h_series.pvd")
        self.p_file = File("data/output/p_series.pvd")
        self._balance_cache: dict[tuple[float, bool, bool], float] = {}
        self._memo_count = 0

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
        self.PRelax = Function(self.V)
        self.Pnow = Function(self.V)
        self.v = TestFunction(self.V)
        self.p.rename("p", "pressure")
        self.gradP = Function(self.W)
    
        self.delta = Function(self.V)
        self.delta_local = Function(self.V)
        self.delta_nonlocal = Function(self.V)
        self.p_old = Function(self.V) #previous pressure for calculating residual during EHL deformation loop

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

    def load_influence_matrix(self):
        self.K = load_npz(self.solver_params.K_matrix_dir)
        # Convert K to a CSR matrix
        self.K = self.K.tocsr()
        print("Loaded influence matrix")

    def split_influence_matrix(self):
        """Split the influece matrix in to local and non-local parts by"""
        """splitting the K matrix in to the diagonal and off-diagonal parts"""
        #find the size of the matrix
        #separate the local and non-local parts
        diags = self.K.diagonal()
        self.K_local = csr_matrix(np.diag(diags))
        self.K_nonlocal = self.K - self.K_local


    def calculate_deformation(self, p, K):
        """Calculate the elastic deformation from the pressure field using the influence matrix."""
        #Input: p as a fenics function, K as a scipy sparse matrix
        #Output: delta as an array, to be loaded in to self.delta or other later
        #The method can be used for any K and p
        ######################################################### NEED TO REVISIT TO CHECK DIMENSIONAL/NONDIMENSIONAL CONSISTENCY
        p_array = p.vector()[:]
        #remove negative pressures from the array
        p_array = np.maximum(p_array, 0)
        delta = K @ p_array * self.material_properties.p0*self.material_properties.Rc / (1e6)
        print(f'Max deformation: {np.max(delta)/self.material_properties.c} m')
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

    # def regularise_contact_separation(self, values: np.ndarray) -> np.ndarray:
    #     """Apply a smooth, pointwise projection that keeps the film non-negative.

    #     Parameters
    #     ----------
    #     values:
    #         Numpy array containing the nodal values of the current film
    #         thickness field.

    #     Returns
    #     -------
    #     numpy.ndarray
    #         The adjusted array with NaNs replaced and, if required,
    #         locally regularised according to
    #         ``0.5 * (h + sqrt(h**2 + s**2))`` where ``s`` is the
    #         regularisation smoothing parameter.
    #     """

    #     if values.size == 0:
    #         return values

    #     mask = np.isfinite(values)
    #     if not np.all(mask):
    #         if np.any(mask):
    #             fill_value = values[mask].max()
    #         else:
    #             fill_value = 0.0
    #         values[~mask] = fill_value

    #     smoothing = self.h_regularisation_param
    #     non_positive = values <= 0.0
    #     if np.any(non_positive):
    #         original_count = int(np.count_nonzero(non_positive))
    #     else:
    #         original_count = 0

    #     values = 0.5 * (values + np.sqrt(values * values + smoothing * smoothing))
    #     np.maximum(values, 0.0, out=values)

    #     if original_count:
    #         print(
    #             "Locally regularised "
    #             f"{original_count} film nodes with smoothing {smoothing:.3e} to ensure positivity."
    #         )

    #     return values



    def regularise_contact_separation(
        self,
        values: np.ndarray,
        smoothing: float | None = None,
        h_min: float | None = None,
    ) -> np.ndarray:
        """
        Smoothly enforce a minimum film thickness.

        Applies:  h_reg = h_min + 0.5 * (x + sqrt(x^2 + s^2)),  x = h - h_min
        which is a smooth approximation to max(h, h_min) with softness 's'.

        Parameters
        ----------
        values : np.ndarray
            Nodal film thickness array (will be returned as a new array).
        smoothing : float, optional
            Smoothness parameter 's' (same units as film). Larger s = softer transition.
            Defaults to self.h_regularisation_param.
        h_min : float, optional
            Minimum permissible film thickness. Defaults to
            getattr(self, "h_min_regularisation", 0.0).

        Returns
        -------
        np.ndarray
            Regularised film thickness array with values >= h_min.
        """
        if values.size == 0:
            return values

        # Defaults from the object if not provided
        if smoothing is None:
            smoothing = float(self.h_regularisation_param)
        if h_min is None:
            h_min = float(getattr(self, "h_min_regularisation", 0.0))

        # Replace non-finite entries with a safe fill (max of finite, else 0)
        vals = values.astype(float, copy=True)
        finite_mask = np.isfinite(vals)
        if not np.all(finite_mask):
            fill_value = vals[finite_mask].max() if np.any(finite_mask) else 0.0
            vals[~finite_mask] = fill_value

        # Count how many nodes are below the floor (for logging only)
        below_floor = vals < h_min
        original_count = int(np.count_nonzero(below_floor))

        # Shift to x = h - h_min, apply smooth max, shift back
        x = vals - h_min
        # numerically stable sqrt(x^2 + s^2)
        vals = h_min + 0.5 * (x + np.hypot(x, smoothing))

        # Guard against tiny numerical undershoot
        np.maximum(vals, h_min, out=vals)

        if original_count:
            print(
                f"Regularised {original_count} film nodes to >= {h_min:.3e} "
                f"with smoothing {smoothing:.3e}."
            )

        return vals



    def update_contact_separation(
        self, eccentricity, HMMState=False, transientState=False, EHLState=False
    ):
        """
        Update the contact separation based on the given eccentricity.
        """
        if EHLState:
            sep_expr = Expression(
                ("(1 - (ecc)*sqrt(1 - pow(x[0],2) - pow(x[1],2))) + delta_local + delta_nonlocal"),
                ecc=eccentricity[2] / self.material_properties.length_ratio,
                delta_local=self.delta_local,
                delta_nonlocal=self.delta_nonlocal,

                element=self.V.ufl_element(),
            )
        else:
            sep_expr = Expression(
                ("(1 - (ecc)*sqrt(1 - pow(x[0],2) - pow(x[1],2)))"),
                ecc=eccentricity[2] / self.material_properties.length_ratio,
                element=self.V.ufl_element(),
            )

        self.h = interpolate(sep_expr, self.V)
        h_local = self.h.vector().get_local()      
        h_local = self.regularise_contact_separation(h_local, smoothing=5e-5, h_min=0.00005)
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
        friction = np.linalg.norm(f_friction[0]) / (self.current_load * self.material_properties.load_mag)
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
        self.friction_coeff = self.tangential_friction_hom / (self.current_load * self.material_properties.load_mag)
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


    def soderfjall_solve(self, HMMState=False, transientState=False):
        alpha_reduction = 0.1
        # find an appropriate beta value by taking a percentage of peak pressure from an uncavitated solve
        self.est_pmag_problem(HMMState=HMMState, transientState=transientState)
        self.fenicssolve()
        p_sol = self.p.vector()[:]
        beta_frac = 0.01
        self.beta = np.max(abs(p_sol)) * beta_frac

        self.alpha.assign(1e4)
        self.initialise_problem(HMMState=HMMState, transientState=transientState)
        while float(self.alpha) > 0.01:
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
        key = (eccentricity, HMMState, transientState)
        if key in self._balance_cache:
            self._memo_count += 1
            return self._balance_cache[key]
        print(f'Solving with eccentricity {eccentricity}')
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
            f'Offset: [{self.material_properties.eccentricity0[2] / self.material_properties.length_ratio: >3.8%}], '
            f'Load: [{" ".join(f"{x: >8.5f}" for x in load)}], '
            f'Force: [{" ".join(f"{x: >8.5f}" for x in self.force)}], '
            f'Difference: [{difference[2]: >3.2e}], '
            f"difference norm: {np.linalg.norm(difference[2]): >9.4e}"
        )
        print(
            f"Maximum pressure MPa: {self.p.vector().max()*self.material_properties.p0/1e6}"
        )
        self._balance_cache[key] = difference[2]
        return difference[2]

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


    def EHL_balance_equation(self, eccentricity, HMMState=False, transientState=False):
        """
        Calculate the difference between the target and actual force vectors.
        """
        ecc = self._as_scalar(eccentricity)
        key = self._cache_key(ecc, HMMState, transientState)

        if key in self._balance_cache:
            self._memo_count += 1
            return self._balance_cache[key]
        
        print(f'Solving with eccentricity {eccentricity}')
        self.material_properties.eccentricity0 = np.array([0.0, 0.0, ecc], dtype=float)
        self.setup_function_spaces()
        self.initialise_bc()
        self.initialise_velocity()
        """"""
        #We need a loop here to converge the deformation
        pressure_residual = 1        
        relaxation_factor = 0.5 #under-relaxation factor for pressure update
        def_relaxation_factor = 0.2
        relaxation_iteration = 0
        pressure_relaxation_tol = 1e-3
        self.prev_pressure_residual = np.inf
        max_relaxation_iterations = 200
        s_old = None
        # We introduce a pressure relaxation loop here to converge the deformation
        print("Starting EHL deformation loop...")
        print(f'Initial pressure residual: {pressure_residual}')
        while pressure_residual > pressure_relaxation_tol and relaxation_iteration < max_relaxation_iterations:
            print(f'Relaxation iteration {relaxation_iteration}, pressure residual {pressure_residual}')
            p_old = self.p.vector().get_local().copy()
            delta_old = self.delta.vector().get_local().copy()

            self.update_contact_separation(
                self.material_properties.eccentricity0,
                HMMState=HMMState,
                transientState=transientState,
                EHLState=True,
            )

            self.soderfjall_solve(HMMState=HMMState, transientState=transientState)
            p_trial = self.p.vector().get_local().copy()

            p_new = (1 - relaxation_factor) * p_old + relaxation_factor * p_trial
            # p_new = np.maximum(p_new, 0) 

            self.p.vector().set_local(p_new)
            self.p.vector().apply("insert")

            delta_local = self.calculate_deformation(self.p, self.K_local)
            delta_nonloc = self.calculate_deformation(self.p, self.K_nonlocal)
            delta_trial = delta_local + delta_nonloc

            s_new = delta_trial - delta_old
            if s_old is not None:
                diff = s_new - s_old
                denom = np.dot(diff, diff)
                if denom > 0:
                    def_relaxation_factor = np.clip(-def_relaxation_factor * np.dot(s_old, diff) / denom, 0.05, 1.3)
            delta_new = delta_old + def_relaxation_factor * s_new
            s_old = s_new.copy()

            self.delta_local.vector()[:] = delta_local
            self.delta_nonlocal.vector()[:] = delta_nonloc
            self.delta.vector()[:] = delta_new

            self.update_contact_separation(
                self.material_properties.eccentricity0,
                HMMState=HMMState,
                transientState=transientState,
                EHLState=True,
            )

            pressure_residual = self.relative_residual(p_new, p_old)

            if pressure_residual > 0.9 * self.prev_pressure_residual if relaxation_iteration > 0 else False:
                relaxation_factor = max(0.5 * relaxation_factor, 0.05)
            else:
                relaxation_factor = min(1.2 * relaxation_factor, 0.8)

            self.prev_pressure_residual = pressure_residual
            print(f'Relaxation iterat {relaxation_iteration}, pressure residual {pressure_residual}, relaxation factor {relaxation_factor}, deflection relaxation factor {def_relaxation_factor}')
            relaxation_iteration += 1

        print(f'Total relaxation iterations: {relaxation_iteration}, final pressure residual: {pressure_residual}')
        """"""
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
            f'Offset: [{self.material_properties.eccentricity0[2] / self.material_properties.length_ratio: >3.8%}], '
            f'Load: [{" ".join(f"{x: >8.5f}" for x in load)}], '
            f'Force: [{" ".join(f"{x: >8.5f}" for x in self.force)}], '
            f'Difference: [{difference[2]: >3.2e}], '
            f"difference norm: {np.linalg.norm(difference[2]): >9.4e}"
        )
        print(
            f"Maximum pressure MPa: {self.p.vector().max()*self.material_properties.p0/1e6}"
        )
        self._balance_cache[key] = difference[2]
        self.export_series('h', relaxation_iteration)
        self.export_series('p', relaxation_iteration)
        return difference[2]

    def solve_loadbalance_EHL(self, HMMState=False, transientState=False):
        """
        Use scipy.optimize.root to find the eccentricity that balances the equation.
        """
        initial_guess = self.material_properties.eccentricity0[2]
        self.main_residuals = []
        tol = self.solver_params.load_balance_rtol
        result = root(
            self.EHL_balance_equation,
            initial_guess,
            args=(HMMState, transientState),
            method="hybr",
            tol=tol,
        )

        if result.success:
            eccentricity_new = self._as_scalar(result.x)
            self.material_properties.eccentricity0 = np.array([0.0, 0.0, eccentricity_new], dtype=float)
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
        

    def solve_loadbalance_bisect(self, HMMState=False, transientState=False):
        """
        Use the secant method via ``scipy.optimize.root_scalar`` to determine the
        eccentricity that balances the equation.
        """
        self._balance_cache.clear()
        self.main_residuals = []
        tol = self.solver_params.load_balance_rtol
        max_ecc = 0.9999 * self.material_properties.length_ratio
        prev_ecc = self.material_properties.eccentricity0[2]
        width = 0.01 * self.material_properties.length_ratio
        max_attempts = 5

        result = None
        residual = None
        fn_calls_count = 0
        for _ in range(max_attempts):
            # Start with a bracket around the previous solution when possible
            a = max(0.0, prev_ecc - width)
            b = min(max_ecc, prev_ecc + width)
            fa = self.EHL_balance_equation(a, HMMState, transientState)
            fb = self.EHL_balance_equation(b, HMMState, transientState)
            fn_calls_count += 2
            print(f'fa: {fa}, fb: {fb}')
            # If the root is not bracketed, expand only the relevant side
            
            while fa * fb > 0:
                print(f'Expanding bracket: [{a}, {b}], fa*fb > 0')
                if fa < 0 and fb < 0:
                    print(f'fa <0, fb <0, expanding lower bound')
                    a = 0.0
                    fa = self.EHL_balance_equation(a, HMMState, transientState)
                    print(f'New fa: {fa}, fb: {fb}')
                    fn_calls_count += 1
                else:
                    print(f'fa >0, fb >0, expanding upper bound')
                    b = min(max_ecc, b + width)
                    fb = self.EHL_balance_equation(b, HMMState, transientState)
                    print(f'fa: {fa}, New fb: {fb}')
                    fn_calls_count += 1

                # # if it still isn't bracketed, fall back to the full range
                # if fa * fb > 0:
                #     a, b = 0.0, max_ecc
                #     fa = self.balance_equation(a, HMMState, transientState)
                #     fb = self.balance_equation(b, HMMState, transientState)

            result = root_scalar(
                self.EHL_balance_equation,
                bracket=[a, b],
                args=(HMMState, transientState),
                method="brentq",
                xtol=1e-10,
            )
            print("Converged:", result.converged)
            print("Reason:", result.flag)
            print("Iterations:", result.iterations)
            print("Function calls:", result.function_calls)
            print("Root:", result.root)
            print(f'Total Function calls {result.function_calls + fn_calls_count}')
            print(f'Total function executions {result.function_calls + fn_calls_count - self._memo_count}')
            residual = self.EHL_balance_equation(result.root, HMMState, transientState)

            if result.converged and abs(residual) <= tol:
                break

            # widen the bracket for the next attempt
            print(f'Solver converged, but residual {residual} exceeds tolerance {tol}')
            prev_ecc = min(max(result.root, 0.0), max_ecc)
            width *= 2

        if not (result and result.converged and abs(residual) <= tol):
            raise RuntimeError(
                f"Load balance solver failed to converge: residual {residual} exceeds tolerance {tol}"
            )
        
        eccentricity_new = result.root
        self.material_properties.eccentricity0 = np.asarray([0, 0, eccentricity_new])
        print(
            f"Optimization successful: {result.converged}, "
            f'Offset: [{" ".join(f"{x / self.material_properties.length_ratio: >7.2%}" for x in self.material_properties.eccentricity0)}]'
        )
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
