#!/usr/bin/env python3
"""Export a DG0 field marking which xi_rot entries were selected by downsampling.

The script mirrors the downsampling stage used in
``analysis/src/optimise_ndtheta_mls.py`` and then builds an ``EHLSolver``
instance following ``run_steady_macroscale_init_penalty.py`` so the mask can
be written to a mesh-based ``.pvd`` file.
"""

from __future__ import annotations

import os
import numpy as np
from scipy.spatial import cKDTree
from fenics import File, Function, FunctionSpace, LogLevel, set_log_level

from dataclasses import asdict
from CONFIGPenalty import material, mesh, solver as solver_params
from coupling.src.functions.coupling_classes import MetaModel3 as SteadyMetaModel
from coupling.src.functions.transient_coupling_classes import (
    MetaModel3 as TransientMetaModel,
)
from macroscale.src.functions.macro_HMM_penalty_transient_EHL import (
    EHLSolver,
    material_parameters,
    mesh_parameters,
    meshfn,
    solver_parameters,
)

set_log_level(LogLevel.ERROR)


def _as_feature_matrix(xi: np.ndarray, transient: bool) -> np.ndarray:
    base_idx = [0, 1, 2, 3, 5, 6, 8, 9]
    if transient:
        base_idx += [11, 12]
    return np.vstack([xi[i] for i in base_idx]).T


def _match_downsample_indices(X_full: np.ndarray, X_down: np.ndarray, tol: float) -> np.ndarray:
    tree = cKDTree(X_full)
    dists, idx = tree.query(X_down, k=1)
    if np.any(dists > tol):
        raise ValueError(
            "Could not map some downsampled points back to xi_rot indices. "
            f"max_dist={dists.max():.3e}, tolerance={tol:.3e}."
        )
    if len(np.unique(idx)) != len(idx):
        raise ValueError("Downsampled points mapped to duplicate full indices.")
    print(f"Mapped downsampled points to xi_rot with max distance {dists.max():.3e}")
    return idx.astype(int)


def _build_binary_mask(xi_rot: np.ndarray, nd_factor: float, transient: bool, match_tol: float) -> np.ndarray:
    X_full = _as_feature_matrix(xi_rot, transient=transient)
    model_cls = TransientMetaModel if transient else SteadyMetaModel
    metamodel = model_cls(Nd_factor=nd_factor)
    _, xi_d = metamodel.build(xi_rot, order=None, init=True, theta=None)

    X_down = _as_feature_matrix(xi_d, transient=transient)
    idx_down = _match_downsample_indices(X_full, X_down, tol=match_tol)

    binary_mask = np.zeros(X_full.shape[0], dtype=float)
    binary_mask[idx_down] = 1.0
    return binary_mask


def _build_solver(output_dir: str) -> EHLSolver:
    solver = EHLSolver(
        meshfn(mesh_parameters(**asdict(mesh))),
        material_parameters(**asdict(material)),
        solver_parameters(**asdict(solver_params)),
        os.path.basename(output_dir),
    )
    solver.initialise_imported_rotation()
    solver.update_contact_separation(
        solver.material_properties.eccentricity0,
        HMMState=False,
        transientState=False,
        EHLState=False,
    )
    return solver


def _populate_dg0(solver: EHLSolver, binary_mask: np.ndarray) -> Function:
    Vdg0 = FunctionSpace(solver.meshin.meshF, "DG", 0)
    indicator = Function(Vdg0)

    if len(binary_mask) == Vdg0.dim():
        indicator.vector()[:] = binary_mask
        print("Assigned binary mask directly to DG0 vector.")
        return indicator

    Vcg1 = FunctionSpace(solver.meshin.meshF, "CG", 1)
    if len(binary_mask) != Vcg1.dim():
        raise ValueError(
            "Binary mask length does not match either DG0 or CG1 dimensions: "
            f"len(mask)={len(binary_mask)}, DG0={Vdg0.dim()}, CG1={Vcg1.dim()}."
        )

    cg_coords = Vcg1.tabulate_dof_coordinates().reshape((-1, solver.meshin.meshF.geometry().dim()))
    dg_coords = Vdg0.tabulate_dof_coordinates().reshape((-1, solver.meshin.meshF.geometry().dim()))
    nearest = cKDTree(cg_coords).query(dg_coords, k=1)[1]
    indicator.vector()[:] = binary_mask[nearest]
    print("Mapped CG1-sized binary mask to DG0 dofs via nearest dof coordinates.")
    return indicator


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Export downsampling indicator on DG0 mesh field")
    parser.add_argument("--transient", action="store_true", help="Use transient downsampling feature layout")
    parser.add_argument("--output_dir", type=str, default="data/output/hmm_job")
    parser.add_argument("--nd_factor", type=float, required=True)
    parser.add_argument("--match_tol", type=float, default=1e-12)
    parser.add_argument("--pvd_name", type=str, default="downsample_indicator_dg0.pvd")
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    xi_path = os.path.join(output_dir, "xi_rot.npy")
    if not os.path.exists(xi_path):
        raise FileNotFoundError(f"Could not find xi_rot file: {xi_path}")

    xi_rot = np.load(xi_path)
    binary_mask = _build_binary_mask(
        xi_rot=xi_rot,
        nd_factor=args.nd_factor,
        transient=args.transient,
        match_tol=args.match_tol,
    )
    np.save(os.path.join(output_dir, "downsample_binary_mask.npy"), binary_mask)

    solver = _build_solver(output_dir)
    indicator = _populate_dg0(solver, binary_mask)

    out_path = os.path.join(output_dir, args.pvd_name)
    File(out_path) << indicator
    print(f"Saved binary mask array: {os.path.join(output_dir, 'downsample_binary_mask.npy')}")
    print(f"Saved DG0 indicator field: {out_path}")


if __name__ == "__main__":
    main()