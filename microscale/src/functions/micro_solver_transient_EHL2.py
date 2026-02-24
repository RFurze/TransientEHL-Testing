"""
Micro-scale transient Reynolds solver in class form.

This refactors **Code 1** into a class hierarchy that mirrors the API of
`MicroSolver_nondim` from **Code 2**:

* **Identical constructor signature** - accepts
  `(mesh, physical_params, solver_settings, k, ht, target_dof=0, preallocated_spaces=None)`.
* **run(Tf, n_steps)** - performs the full steady → transient procedure that
  previously lived in *Code 1's* main-script, storing the field variables from
  the **final** time-step.
* **spatial_homogenisation()** - returns `(Qx, Qy, P̄, s̄, tau_x, tau_y)` evaluated with the
  final pressure/film fields using the same non-dimensional scalings as
  `MicroSolver_nondim`.
"""

from __future__ import annotations

# import logging
import os
import sys
import io
import time
from contextlib import contextmanager
from math import pi, sin, cos
from typing import Dict, Tuple, Optional, List

import numpy as np
import ngsolve
from ngsolve import *
from ngsolve.solvers import *
import xml.etree.ElementTree as ET

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     filename="simulation_log.log",
#     filemode="w",
# )
# logger = logging.getLogger(__name__)


@contextmanager
def capture_output(stream_name: str = "stdout"):
    """Context manager that captures spam from Newton solver."""
    old_stream = getattr(sys, stream_name)
    buf = io.StringIO()
    setattr(sys, stream_name, buf)
    try:
        yield buf
    finally:
        setattr(sys, stream_name, old_stream)


def _write_pvd_file(
    output_dir: str, pvd_filename: str, file_times: list[tuple[str, float]]
) -> None:
    vtkfile = ET.Element(
        "VTKFile", type="Collection", version="0.1", byte_order="LittleEndian"
    )
    collection = ET.SubElement(vtkfile, "Collection")
    for fname, t in file_times:
        ET.SubElement(collection, "DataSet", timestep=str(t), file=fname)

    ET.ElementTree(vtkfile).write(
        os.path.join(output_dir, pvd_filename), encoding="utf-8", xml_declaration=True
    )


# Dowson-Higginson compressibility (dimensional)
C1_DH = 0.59e9
C2_DH = 1.34


def dh_ratio(p):
    """rho/rho₀ compressibility ratio; returns 1 when gauge-pressure ≤ 0."""
    return IfPos(p, (C1_DH + C2_DH * p) / (C1_DH + p), 1.0)


def roelands(p):
    """
    Roelands viscosity model in multiplicative/log form:

        eta = eta_p * (eta00/eta_p) ** ((1 + max(p,0)/p_r)**z_capped)

    where z_capped := min((1 + max(p,0)/p_r)**z, expo_cap).
    """

    # --- parameters (Pa·s, Pa, dimensionless) ---
    eta_p = 6.31e-5  # Pa·s  (limiting viscosity at high pressure)
    eta00 = 1.0e-2  # Pa·s  (zero-pressure viscosity)
    p_r = 0.5e8  # Pa    (Roelands reference pressure; POSITIVE)
    z = 0.548  # Roelands exponent
    expo_cap = 10.0  # cap on exponent to avoid runaway growth

    # clamp negative gauge pressures to zero
    p_pos = IfPos(p, p, 0.0)

    # compute exponent and cap it
    expo = (1.0 + p_pos / p_r) ** z
    expo_c = IfPos(expo_cap - expo, expo, expo_cap)  # if expo < cap use expo else cap

    # log/exp form for numerical stability
    base_log = log(eta00 / eta_p)
    ln_eta = log(eta_p) + expo_c * base_log
    return exp(ln_eta)


class micro_PhysicalParameters:
    def __init__(
        self,
        Ux,
        Uy,
        eta0,
        rho0,
        alpha,
        beta,
        beta_fraction,
        xmax,
        ymax,
        p0,
        dpdx,
        dpdy,
        H0,
        HT,
        PT,
        Tend,
    ):
        self.Ux = 2 * Ux  # Account for incoming velocity being 1/2 already
        self.Uy = 2 * Uy
        self.eta0 = eta0
        self.rho0 = rho0
        self.alpha = alpha
        self.beta = beta
        self.beta_fraction = beta_fraction
        self.xmax = xmax
        self.ymax = ymax
        self.p0 = p0
        self.dpdx = dpdx
        self.dpdy = dpdy
        self.H0 = H0
        self.HT = HT
        self.PT = PT
        self.Tend = Tend


class micro_SolverSettings:
    def __init__(
        self,
        newton_damping: float,
        max_iterations: int,
        error_tolerance: float,
        alpha_reduction_factor: float,
        alpha_threshold: float,
    ):
        self.newton_damping = newton_damping
        self.max_iterations = max_iterations
        self.error_tolerance = error_tolerance
        self.alpha_reduction_factor = alpha_reduction_factor
        self.alpha_threshold = alpha_threshold


# Main Microscale solver class


class MicroTransientSolver_nondim:
    def __init__(
        self,
        mesh: Mesh,
        physical_params: micro_PhysicalParameters,
        solver_settings: micro_SolverSettings,
        k: int,
        k_spring: float,
        ht,
        target_dof: int = 0,
        export_vtk: bool = False,
        output_dir: str | None = None,
        preallocated_spaces: Optional[Dict[str, ngsolve.FESpace]] = None,
    ) -> None:
        self.mesh = mesh
        self.params = physical_params
        self.settings = solver_settings
        self.k = k
        self.k_spring = k_spring
        self.ht_cb = ht
        self.target_dof = target_dof
        self.export_vtk = export_vtk
        self.output_dir = output_dir or "output"

        if self.export_vtk:
            os.makedirs(self.output_dir, exist_ok=True)
            self._file_times: list[tuple[str, float]] = []

        # --- Finite-element spaces (optionally re-used between solver clones)
        if preallocated_spaces is None:
            self.V = Compress(H1(mesh, order=k))
            self.lag_fb = Compress(
                Periodic(H1(mesh, order=1, definedon=mesh.Boundaries("bottom|top")))
            )
            self.lag_lr = Compress(
                Periodic(H1(mesh, order=1, definedon=mesh.Boundaries("left|right")))
            )

            # Calculate references of corner nodes and set them as fixed
            nverts_edge = int(round(np.sqrt(self.mesh.nv)))
            n = nverts_edge - 1
            freedofs = [0, n * nverts_edge, n, n * nverts_edge + n]
            for dof in freedofs:
                self.V.FreeDofs()[dof] = False
            self.X = self.V * self.lag_fb * self.lag_lr
        else:
            self.V = preallocated_spaces["V"]
            self.lag_fb = preallocated_spaces["lag_fb"]
            self.lag_lr = preallocated_spaces["lag_lr"]
            self.X = preallocated_spaces["X"]

        self.u, self.lam, self.gamma = self.X.TrialFunction()
        self.v, self.mu, self.nu = self.X.TestFunction()
        self.gfu = GridFunction(self.X)

        self._setup_scaling()
        self.set_corner_constraints(None, None, 0)

        self._final_pressure: Optional[GridFunction] = None
        self._final_film: Optional[CoefficientFunction] = None
        self._final_alpha: Optional[float] = None
        self._final_time: float = 0.0

    def set_corner_constraints(self, p_guess, p_last, t) -> None:
        # Input t should be nondimensional
        p_ref_nd = self.p_ref_nd
        if self.params.Tend is None or self.params.Tend == 0:
            pt_ref_nd = 0
        else:
            pt_ref_nd = (
                (self.p_ref_nd1 - self.p_ref_nd) * t / (self.params.Tend / self.t0)
            )
        dpx_nd = self.dpx_nd
        dpy_nd = self.dpy_nd

        nverts_edge = int(round(np.sqrt(self.mesh.nv)))
        n = nverts_edge - 1
        freedofs = [0, n * nverts_edge, n, n * nverts_edge + n]
        if p_guess is not None:
            self.gfu.components[0].vec.data = p_guess.vec
        elif p_last is not None:
            self.gfu.components[0].vec.data = p_last.vec
        else:
            self.gfu.components[0].vec[:] = pt_ref_nd + p_ref_nd
        self.gfu.components[0].vec[freedofs[0]] = pt_ref_nd + p_ref_nd
        self.gfu.components[0].vec[freedofs[1]] = pt_ref_nd + p_ref_nd + dpx_nd
        self.gfu.components[0].vec[freedofs[2]] = pt_ref_nd + p_ref_nd + dpy_nd
        self.gfu.components[0].vec[freedofs[3]] = pt_ref_nd + p_ref_nd + dpx_nd + dpy_nd

        # print(f't{t}, p_ref_nd {p_ref_nd}, pt_ref_nd {pt_ref_nd}, corner 0 {pt_ref_nd + p_ref_nd}, corner 1 {pt_ref_nd + p_ref_nd + dpx_nd}, corner 2 {pt_ref_nd + p_ref_nd + dpy_nd}, corner 3 {pt_ref_nd + p_ref_nd + dpx_nd + dpy_nd}')

    def run(self, Tf: float, n_steps: int = 10) -> None:
        """Execute the full transient simulation up to *Tf* seconds.

        Results (pressure/film/alpha) for the **final** time step are cached inside
        the instance and subsequently used by :py:meth:`spatial_homogenisation`.
        """
        # Nondimensional time
        Tf = Tf / self.t0
        dt = Tf / n_steps
        T = 0.0

        p_hist: List[GridFunction] = []
        h_hist: List[CoefficientFunction] = []
        p_last_tr: Optional[GridFunction] = None
        h_last_tr: Optional[CoefficientFunction] = None

        # Settings for the per-timestep fixed-point on h_d = k * Pst
        HD_MAX_OUTER_IT = getattr(self.settings, "hd_max_outer_it", 25)
        HD_TOL          = getattr(self.settings, "hd_rel_tol", self.settings.error_tolerance)
        HD_RELAX        = getattr(self.settings, "hd_relax", 1.0)  # 0<η≤1 optional damping

        # Seed h_d with input mean pressure (uniform)
        hd_nd_prev = (self.params.p0 / self.k_spring) / self.params.H0

        # ----------------------------------------------------------------
        #  Time loop
        # ----------------------------------------------------------------
        step = 0
        alpha_current = self.params.alpha
        while T < Tf + 1e-12:
            # 1) Steady predictor at this time to get alpha + initial guess
            if step == 0 or p_last_tr is None:
                (
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    p_steady,
                    h_steady,
                    _,
                    _,
                    alpha_final,
                ) = self._solve_micro_step(
                    T,
                    dt,
                    p_last=None,
                    h_last=None,
                    p_last2=None,
                    h_last2=None,
                    include_transient_terms=False,
                    continuation=True,
                    fixed_alpha=None,
                    p_guess=None,
                    hd_nd=hd_nd_prev,
                )
                p_guess_for_tr = p_steady
                alpha_current = alpha_final
            else:
                alpha_final = alpha_current
                p_guess_for_tr = p_last_tr

            # 2) Choose BE/BDF2 histories
            if len(p_hist) >= 2:
                p_last, h_last = p_hist[-1], h_hist[-1]
                p_last2, h_last2 = p_hist[-2], h_hist[-2]
            elif len(p_hist) == 1:
                p_last, h_last = p_hist[-1], h_hist[-1]
                p_last2 = h_last2 = None
            else:
                p_last = p_last2 = h_last = h_last2 = None

            # 3) Per-timestep fixed-point on h_d with transient terms ON
            hd_nd   = hd_nd_prev
            p_prev_outer: Optional[GridFunction] = None
            for oit in range(HD_MAX_OUTER_IT):
                (
                    _,
                    _,
                    _,
                    h_curr,
                    _,
                    _,
                    p_curr,
                    _,
                    _,
                    _,
                    _,
                ) = self._solve_micro_step(
                    T,
                    dt,
                    p_last,
                    h_last,
                    p_last2,
                    h_last2,
                    include_transient_terms=True,
                    continuation=False,
                    fixed_alpha=alpha_final,
                    p_guess=p_guess_for_tr,
                    hd_nd=hd_nd,
                )

                if p_prev_outer is not None:
                    res = Norm(p_curr.vec - p_prev_outer.vec) / max(1.0, Norm(p_prev_outer.vec))
                    # print(f"[transient hd-loop] step={step} oit={oit} res={res:.3e} hd_nd={hd_nd:.6e}")
                    if res < HD_TOL:
                        break
                p_prev_outer = GridFunction(p_curr.space); p_prev_outer.vec.data = p_curr.vec

                # Update h_d from homogenised microscale pressure
                Pst_dim = self._mean_pressure_dimensional(p_curr)  # Pa
                hd_nd_new = (np.max(Pst_dim,0) / self.k_spring) / self.params.H0
                hd_nd = (1.0 - HD_RELAX) * hd_nd + HD_RELAX * hd_nd_new

                # Update guess for next inner solve to accelerate convergence
                p_guess_for_tr = p_curr

            # 4) Commit fields & bookkeeping
            p_last_tr = p_curr
            h_last_tr = h_curr
            self.post_process(step, T, p_last_tr, h_last_tr)

            if step == 0:
                self._final_pressure = p_last_tr
                self._final_film = h_last_tr
                self._final_alpha = alpha_final

            p_hist.append(p_last_tr); h_hist.append(h_last_tr)
            if len(p_hist) > 2:
                p_hist.pop(0); h_hist.pop(0)

            hd_nd_prev = hd_nd
            T += dt
            step += 1

        sys.stdout.flush()
        if self.export_vtk:
            _write_pvd_file(self.output_dir, "solution_transient.pvd", self._file_times)
            print(f"Wrote {len(self._file_times)} VTK files to {self.output_dir}")
        self._final_pressure = p_last_tr
        self._final_film = h_last_tr
        self._final_alpha = alpha_final
        self._final_time = T - dt

    def calculate_maxmin(self, k):
        # k is a field
        # returns the dimensionless max/min of k as a tuple
        V1 = H1(self.mesh, order=1)
        out1 = GridFunction(V1)
        out1.Set(k)
        vals = out1.vec.FV().NumPy()
        k_max = vals.max()
        k_min = vals.min()
        return k_max, k_min

    def spatial_homogenisation(
        self,
    ) -> Tuple[
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
    ]:
        """Return (⟨Qx⟩, ⟨Qy⟩, ⟨P⟩, ⟨s⟩, τ_x, τ_y) for the final time step."""
        if self._final_pressure is None or self._final_film is None:
            raise RuntimeError("run() must be executed before homogenisation.")

        pressure = self._final_pressure
        h = self._final_film
        Ux, Uy = self.Ux_nd, self.Uy_nd

        grad_px = GridFunction(self.V)
        grad_py = GridFunction(self.V)
        grad_px.Set(grad(pressure)[0])
        grad_py.Set(grad(pressure)[1])

        f = IfPos(
            pressure,
            1,
            IfPos(
                -pressure - self.beta_nd,
                0,
                1
                - 2 * (pressure / self.beta_nd) ** 3
                - 3 * (pressure / self.beta_nd) ** 2,
            ),
        )
        s = (f + 1e-2) / (
            1 + 1e-2
        )  ######################################################### Shouldnt hard code alpha!!!

        roelands_eta = roelands(pressure * self.p0) / self.params.eta0
        qx = Ux * h - self.omega * h**3 * grad_px / (s*roelands_eta)
        qy = Uy * h - self.omega * h**3 * grad_py / (s*roelands_eta)
        Qx = self.qx0 * Integrate(qx, self.mesh)
        Qy = self.qy0 * Integrate(qy, self.mesh)
        Pst = Integrate(pressure, self.mesh) * self.p0
        Fst = Integrate(f, self.mesh)

        # Calculate max and min p, q, F
        p_max, p_min = self.calculate_maxmin(pressure)
        h_max, h_min = self.calculate_maxmin(h)
        p_max, p_min = p_max * self.p0, p_min * self.p0
        h_max, h_min = h_max * self.params.H0, h_min * self.params.H0
        Qx_max, Qx_min, Qy_max, Qy_min, F_max, F_min = 0, 0, 0, 0, 0, 0  # Placeholders

        tau_x_expr = s * roelands_eta * Ux / h + self.omega * h * grad_px / 3
        tau_y_expr = s * roelands_eta * Uy / h + self.omega * h * grad_py / 3
        taust_x = self.tau0 * Integrate(tau_x_expr, self.mesh)
        taust_y = self.tau0 * Integrate(tau_y_expr, self.mesh)

        return (
            float(Qx),
            float(Qy),
            float(Pst),
            float(Fst),
            float(taust_x),
            float(taust_y),
            float(p_max),
            float(p_min),
            float(h_max),
            float(h_min),
            float(Qy_max),
            float(Qy_min),
            float(F_max),
            float(F_min),
        )

    def post_process(
        self,
        step_id: int,
        time_value: float,
        pressure: GridFunction,
        film: CoefficientFunction,
    ) -> None:
        """
        Write *pressure* and film thickness for the current step to
        `<output_dir>/step_<NNNN>.vtu` **and** remember the name/time so
        we can emit a single `.pvd` index after the run completes.
        """
        if not self.export_vtk:
            return  # exit if disabled

        film_gf = GridFunction(self.V)
        film_gf.Set(film)

        FluidFraction_gf = GridFunction(self.V)
        FluidFraction_gf.Set(self._fluid_fraction(pressure))

        fname_base = f"step_{step_id:04d}"
        VTKOutput(
            self.mesh,
            coefs=[pressure, film_gf, FluidFraction_gf],
            names=["pressure", "film", "fluid fraction"],
            filename=os.path.join(self.output_dir, fname_base),
            subdivision=2,
        ).Do()

        # remember for meta-file
        self._file_times.append((f"{fname_base}.vtu", time_value))

    def post_process_steady(
        self, pressure: GridFunction, film: CoefficientFunction
    ) -> None:
        """
        Write *pressure* and *film thickness* for the current step to
        `<output_dir>/step_<NNNN>.vtu` **and** remember the name/time so
        we can emit a single `.pvd` index after the run completes.
        """
        if not self.export_vtk:
            return  # early-out if disabled

        # Convert h(x,y) expression → GridFunction so VTKOutput can see it
        film_gf = GridFunction(self.V)
        film_gf.Set(film)

        FluidFraction_gf = GridFunction(self.V)
        FluidFraction_gf.Set(self._fluid_fraction(pressure))

        VTKOutput(
            self.mesh,
            coefs=[pressure, film_gf, FluidFraction_gf],
            names=["pressure", "film", "fluid fraction"],
            filename=os.path.join(self.output_dir, "steady"),
            subdivision=2,
        ).Do()

    def _setup_scaling(self):
        """Set up non dimensionalisation for the steady case at t=0."""
        self.p0 = max(
            abs(self.params.p0),
            abs(self.params.p0 + self.params.dpdx * self.params.xmax),
            abs(self.params.p0 + self.params.dpdy * self.params.ymax),
            abs(
                self.params.p0
                + self.params.dpdx * self.params.xmax
                + self.params.dpdy * self.params.ymax
            ),
            abs(self.params.p0 + self.params.Tend * self.params.PT),
            abs(
                self.params.p0
                + self.params.Tend * self.params.PT
                + self.params.dpdx * self.params.xmax
                + self.params.dpdy * self.params.ymax
            ),
            abs(
                self.params.p0
                + self.params.Tend * self.params.PT
                + self.params.dpdx * self.params.xmax
            ),
            abs(
                self.params.p0
                + self.params.Tend * self.params.PT
                + self.params.dpdy * self.params.ymax
            ),
            1,
        )
        U_mag = abs(self.params.Ux)
        if U_mag == 0:
            raise ValueError("Ux must be non-zero for transient scaling")
        self.t0 = 2 * self.params.xmax / (U_mag + 1e-12)
        # print(f't0 = {self.t0}')

        self.beta_nd = self.params.beta / self.p0  # nondimensional β
        self.omega = (
            self.params.H0**2
            * self.p0
            / (6 * self.params.eta0 * self.params.xmax * abs(self.params.Ux))
        )

        self.dpx = self.params.dpdx * self.params.xmax
        self.dpy = self.params.dpdy * self.params.ymax

        self.dpx_nd = self.dpx / self.p0  # nondimensional pressure offset
        self.dpy_nd = self.dpy / self.p0  # nondimensional pressure offset

        # pr is what we use as a reference pressure scale
        # p_ref is the pressure used for constraints
        # set up a p_ref without time dependence - this will be added in during the run
        self.p_ref_0 = self.params.p0 - 0.5 * (self.dpx + self.dpy)
        self.p_ref_1 = self.p_ref_0 + self.params.PT * self.params.Tend
        
        def min_corner(p_ref):
            return min(
                p_ref,
                p_ref + self.dpx,
                p_ref + self.dpy,
                p_ref + self.dpx + self.dpy,
            )

        p_min0 = min_corner(self.p_ref_0)
        p_min1 = min_corner(self.p_ref_1)
        p_min_all = min(p_min0, p_min1)
        safety = 1e-8  # or e.g. 1e-8 * self.p0 if you want strict > -beta
        if p_min_all <= -self.params.beta:
            shift = -self.params.beta + safety - p_min_all
        else:
            shift = 0.0

        self.p_ref_0 += shift
        self.p_ref_1 += shift

        # We use guage pressure of -beta here for consistency with the macroscale
        # If we used 0 as the guage pressure, we would never get cavitation in the parallel plate case when input pressure < 0.
        # print(f'beta: {-self.params.beta}')
        # if self.p_ref_m > -self.params.beta:
        #     self.p_ref = self.p_ref_0
        # else:
        #     self.p_ref = self.p_ref_0 - self.p_ref_m

        # if self.p_ref_m1 > -self.params.beta:
        #     self.p_ref1 = self.p_ref_1
        # else:
        #     self.p_ref1 = self.p_ref_1 - self.p_ref_m1

        # Reference pressures at the start and end times to compare
        # self.p_ref_nd = self.p_ref / self.p0
        # self.p_ref_nd1 = self.p_ref1 / self.p0
        self.p_ref_nd  = self.p_ref_0 / self.p0
        self.p_ref_nd1 = self.p_ref_1 / self.p0

        self.HT_nd = self.params.HT * self.t0 / self.params.H0
        self.PT_nd = self.params.PT * self.t0 / self.p0

        self.Ux_nd = self.params.Ux / abs(self.params.Ux)
        self.Uy_nd = self.params.Uy / abs(self.params.Ux)

        self.qx0 = np.abs(self.params.Ux) * self.params.H0 / 2
        self.qy0 = self.qx0

        self.tau0 = self.params.eta0 * self.params.Ux / self.params.H0

        # print(f"[micro scaling] p0={self.params.p0:.3e}, PT={self.params.PT:.3e}, Tend={self.params.Tend}")
        # print(f"  dpx={self.dpx:.3e}, dpy={self.dpy:.3e}")
        # print(f"  p_ref0={self.p_ref_0:.3e}, p_ref1_raw={self.p_ref_1:.3e}")
        # print(f"  p_ref_m={self.p_ref_m:.3e}, p_ref_m1={self.p_ref_m1:.3e}")
        # print(f"  p_ref={self.p_ref:.3e}, p_ref1={self.p_ref1:.3e}")

    #  Core micro solver

    def _solve_micro_step(
        self,
        T: float,
        dt: float,
        p_last: Optional[GridFunction],
        h_last: Optional[CoefficientFunction],
        p_last2: Optional[GridFunction],
        h_last2: Optional[CoefficientFunction],
        *,
        include_transient_terms: bool,
         continuation: bool,
         fixed_alpha: Optional[float],
         beta_continuation: bool = False,
         beta_scale_init: float = 1000.0,
        p_guess: Optional[GridFunction],
        hd_nd: Optional[float] = None,
    ):
        """Mostly verbatim port of `run_micro` from Code 1 (but nondim)."""
        start = time.time()

        # Film thickness at current time (nondim!)
        h = (
            self.ht_cb(
                x,
                y,
                self.params.xmax,
                self.params.ymax,
                self.params.H0,
                self.params.HT,
                T * self.t0,
                self.params.Ux,
                self.params.Uy,
            )
            / self.params.H0
        )
        # Add uniform "spring" deflection h_d = k * Pst (nondimensionalised by H0)
        if hd_nd is not None and hd_nd != 0.0:
            h = h + hd_nd

        # Assemble once-per-Newton (closure so we can rebuild with different β_eff and α)
        def newton_solve(alpha_val: float, beta_eff_nd: float):
            u = self.u
            v = self.v
            lam = self.lam
            mu = self.mu
            gamma = self.gamma
            nu = self.nu
            gfu = self.gfu
            Ux = self.Ux_nd
            Uy = self.Uy_nd
 
            # Cavitation with effective β used for continuation;
            # note: nondimensionalisation remains based on *target* self.beta_nd.
            FluidFraction = IfPos(
                u,
                1,
                IfPos(
                    -u - beta_eff_nd,
                    0,
                    1 - 2 * (u / beta_eff_nd) ** 3 - 3 * (u / beta_eff_nd) ** 2,
                ),
            )
            # Need a switch in config to turn on/off the rheology
            dh = dh_ratio(u * self.p0)
            # dh = 1
            roelands_eta = roelands(u * self.p0) / self.params.eta0
            # roelands_eta = 1
            eta = roelands_eta * (FluidFraction + alpha_val) / (1 + alpha_val)
            rho = dh * (FluidFraction + alpha_val) / (1 + alpha_val)

            a = BilinearForm(self.X)
            # Reynolds equation
            a += (
                rho
                * self.omega
                * (h**3 / eta)
                * (grad(u)[0] * grad(v)[0] + grad(u)[1] * grad(v)[1])
                * dx
            )
            a += -rho * (Ux * h * grad(v)[0] + Uy * h * grad(v)[1]) * dx

            # Boundary constraints using Lagrange multipliers
            a += (u * mu + v * lam) * ds("bottom") - (u * mu + v * lam) * ds("top")
            a += (u * nu + v * gamma) * ds("left") - (u * nu + v * gamma) * ds("right")
            a += self.dpy_nd * mu * ds("top") + self.dpx_nd * nu * ds("right")

            # Transient BDF contributions
            if (
                include_transient_terms
                and T > 0.0
                and p_last is not None
                and h_last is not None
            ):
                FluidFraction_last = IfPos(
                    p_last,
                    1,
                    IfPos(
                        -p_last - beta_eff_nd,
                        0,
                        1 - 2 * (p_last / beta_eff_nd) ** 3 - 3 * (p_last / beta_eff_nd) ** 2,
                    ),
                )

                dh_last = dh_ratio(p_last * self.p0)
                # dh_last = 1
                rho_last = dh_last * (FluidFraction_last + alpha_val) / (1 + alpha_val)

                if p_last2 is not None and h_last2 is not None:
                    FluidFraction_last2 = IfPos(
                        p_last2,
                        1,
                        IfPos(
                            -p_last2 - beta_eff_nd,
                            0,
                            1 - 2 * (p_last2 / beta_eff_nd) ** 3 - 3 * (p_last2 / beta_eff_nd) ** 2,
                        ),
                    )
                    dh_last2 = dh_ratio(p_last2 * self.p0)
                    # dh_last2 = 1
                    rho_last2 = (
                        dh_last2 * (FluidFraction_last2 + alpha_val) / (1 + alpha_val)
                    )
                    alf0, alf1, alf2 = (1.5, -2.0, 0.5)
                else:
                    rho_last2 = None
                    alf0, alf1, alf2 = (1.0, -1.0, 0.0)

                a += (alf0 / dt) * rho * h * v * dx
                a += (alf1 / dt) * rho_last * h_last * v * dx
                if rho_last2 is not None:
                    a += (alf2 / dt) * rho_last2 * h_last2 * v * dx

            self.set_corner_constraints(p_guess, p_last, T)
            gfu = self.gfu

            # Newton solve -------------------------------------------------
            with capture_output("stdout") as buf:
                Newton(
                    a,
                    gfu,
                    freedofs=self.X.FreeDofs(),
                    maxit=self.settings.max_iterations,
                    maxerr=self.settings.error_tolerance,
                    inverse="pardiso",
                    dampfactor=self.settings.newton_damping,
                    printing=True,
                )
            return buf.getvalue()

        # --- Continuation strategy selection ---
        target_alpha = 1.0e-2 if fixed_alpha is None else float(fixed_alpha)
        if continuation and beta_continuation:
            # β-continuation: ramp β_eff from (beta_scale_init * β_target) → β_target
            beta_factor = beta_scale_init
            # Prefer an explicit β step factor if provided; else reuse alpha_reduction_factor safely
            beta_step_div = getattr(self.settings, "beta_reduction_factor", None)
            if beta_step_div is None or beta_step_div <= 1.0:
                # Fall back to reciprocal of alpha_reduction_factor if that was <1
                arf = getattr(self.settings, "alpha_reduction_factor", 0.5)
                beta_step_div = 1.0 / max(arf, 1e-6)  # e.g. arf=0.5 → ÷2

            while beta_factor > 1.0 + 1e-12:
                beta_eff_nd = beta_factor * self.beta_nd
                _ = newton_solve(target_alpha, beta_eff_nd)
                beta_factor = max(1.0, beta_factor / beta_step_div)
            # Final solve exactly at target β
            _ = newton_solve(target_alpha, self.beta_nd)
            alpha_final = target_alpha
        else:
            # No continuation: single solve with fixed α and target β
            _ = newton_solve(target_alpha, self.beta_nd)
            alpha_final = target_alpha

        duration = time.time() - start
        pressure = self.gfu.components[0]
        film = h
        # pstprint = Integrate(pressure, self.mesh) * self.p0
        # print(f'{pstprint:.6e}')
        return (
            duration,
            (0.0, 0.0),  # Qx, Qy - not needed at this level
            (0.0, 0.0),  # boundary fluxes - not needed
            film,
            0.0,  # Pst placeholder
            0.0,  # Fst placeholder
            pressure,
            film,
            0.0,
            0.0,
            alpha_final,
        )

    def _mean_pressure_dimensional(self, p_gf: GridFunction) -> float:
        """
        Homogenised microscale pressure Pst (dimensional, Pa) defined as the
        spatial average of the nondimensional pressure multiplied by p0.
        """
        pbar_nd = Integrate(p_gf, self.mesh)
        return float(pbar_nd * self.p0)


    def run_steady(self) -> None:
        """
        Solve a single, steady Reynolds problem with **fixed α=1e-2** and
        **β-continuation** (β_eff: 1000×β_target → β_target). No transient terms.
        """
        T = 0.0   # nondimensional time
        dt = 1.0  # any positive number (unused for include_transient_terms=False)

        # --- Fixed-point loop over h_d = k * Pst ---
        MAX_OUTER_IT = 50
        TOL = self.settings.error_tolerance
        # initial guess: Pst ≈ input mean pressure
        hd_nd = (self.params.p0 / self.k_spring) / self.params.H0
        p_prev: Optional[GridFunction] = None
        alpha_final = None

        for it in range(MAX_OUTER_IT):
            # print(f'[steady hd-loop] Iteration {it}')
            (
                _,
                _,
                _,
                h_steady,
                _,
                _,
                p_steady,
                _,
                _,
                _,
                alpha_final,
            ) = self._solve_micro_step(
                T,
                dt,
                p_last=None,
                h_last=None,
                p_last2=None,
                h_last2=None,
                include_transient_terms=False,   # steady
                continuation=True,               # use continuation
                fixed_alpha=1.0e-2,              # fixed α
                beta_continuation=True,          # turn on β-continuation
                beta_scale_init=1000.0,          # β_eff starts at 1000×β
                p_guess=None,
                hd_nd=hd_nd,                     
            )

            # Convergence check on pressure field
            if p_prev is not None:
                res = Norm(p_steady.vec - p_prev.vec) / max(1.0, Norm(p_prev.vec))
                # print(f"[steady hd-loop] it={it}  res={res:.3e}  hd_nd={hd_nd:.6e}")
                if res < TOL:
                    # print(f'[steady hd-loop] Converged after {it} iterations. res={res:.3e}, tol={TOL:.3e}')
                    break
            p_prev = GridFunction(p_steady.space)
            p_prev.vec.data = p_steady.vec

            # Update Pst and h_d
            Pst_dim = self._mean_pressure_dimensional(p_steady)  # Pa

            # Pst_dim_prev = Pst_dim
            hd_nd = (np.max(Pst_dim,0) / self.k_spring) / self.params.H0

        # Cache final fields and write once
        self._final_pressure = p_steady
        self._final_film = h_steady
        self._final_alpha = alpha_final if alpha_final is not None else self.params.alpha
        self._final_time = 0.0
        self.post_process(step_id=0, time_value=0.0, pressure=p_steady, film=h_steady)

    def _fluid_fraction(self, p: GridFunction | CoefficientFunction):
        beta = self.beta_nd
        return IfPos(
            p,
            1,
            IfPos(
                -p - beta,
                0,
                1 - 2 * (p / beta) ** 3 - 3 * (p / beta) ** 2,
            ),
        )
