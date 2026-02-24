"""
Legacy FEniCS (dolfin) parameter sweep with multiprocessing + verification.

Adds:
- Each task returns (id, pid, avg_u)
- Prints per-PID task counts
- Saves a CSV mapping task->PID
- Optional plot of PID assignment vs task index

Run:
    python fenics_param_sweep_mp.py
"""

import os
import multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt


def solve_one_task(task):
    """
    Worker: solve PDE and return (task_id, pid, avg_u).
    """
    pid = os.getpid()

    # Import inside worker to avoid issues when multiprocessing spawns processes.
    from dolfin import (
        UnitSquareMesh,
        FunctionSpace,
        TrialFunction,
        TestFunction,
        Function,
        Constant,
        Expression,
        DirichletBC,
        dot,
        grad,
        dx,
        assemble,
        solve,
        set_log_level,
        LogLevel,
    )

    # Quiet solver spam (optional)
    set_log_level(LogLevel.ERROR)

    task_id = int(task["id"])

    # Unpack parameters
    k_val = float(task["k"])
    A = float(task["A"])
    x0 = float(task["x0"])
    y0 = float(task["y0"])
    sigma = float(task["sigma"])
    n = int(task["n"])

    # Coarse mesh + CG1 space
    mesh = UnitSquareMesh(n, n)
    V = FunctionSpace(mesh, "CG", 1)

    u = TrialFunction(V)
    v = TestFunction(V)

    k = Constant(k_val)

    # Gaussian source term
    f = Expression(
        "A*exp(-((x[0]-x0)*(x[0]-x0) + (x[1]-y0)*(x[1]-y0)) / (2.0*sigma*sigma))",
        degree=3,
        A=A,
        x0=x0,
        y0=y0,
        sigma=sigma,
    )

    bc = DirichletBC(V, Constant(0.0), "on_boundary")

    a = k * dot(grad(u), grad(v)) * dx
    L = f * v * dx

    uh = Function(V)
    solve(a == L, uh, bc)

    vol = assemble(Constant(1.0) * dx(domain=mesh))
    avg_u = assemble(uh * dx) / vol

    return (task_id, pid, float(avg_u))


def make_200_tasks(seed=123, n=16):
    rng = np.random.default_rng(seed)
    tasks = []
    for i in range(200):
        tasks.append(
            {
                "k": rng.uniform(0.1, 2.0),
                "A": rng.uniform(0.5, 5.0),
                "x0": rng.uniform(0.15, 0.85),
                "y0": rng.uniform(0.15, 0.85),
                "sigma": rng.uniform(0.05, 0.20),
                "n": n,
                "id": i,
            }
        )
    return tasks


def main(num_procs=4):
    tasks = make_200_tasks(seed=123, n=16)

    ctx = mp.get_context("spawn")

    with ctx.Pool(processes=num_procs) as pool:
        # imap preserves input order, so returned tuples align with tasks order
        results = list(pool.imap(solve_one_task, tasks, chunksize=1))

    # Unpack results
    task_ids = np.array([r[0] for r in results], dtype=int)
    pids = np.array([r[1] for r in results], dtype=int)
    outputs = np.array([r[2] for r in results], dtype=float)

    # Verify ordering
    if not np.all(task_ids == np.arange(len(tasks))):
        raise RuntimeError("Task IDs are not in expected order (0..199).")

    # ---- Verification prints
    unique_pids, counts = np.unique(pids, return_counts=True)
    print("\n=== Multiprocessing verification ===")
    print(f"Requested processes: {num_procs}")
    print(f"Unique worker PIDs observed: {len(unique_pids)}")
    print("Tasks per PID:")
    for pid, c in sorted(zip(unique_pids.tolist(), counts.tolist()), key=lambda x: x[0]):
        print(f"  PID {pid}: {c} tasks")

    print("\nFirst 20 task assignments (task_id -> PID):")
    for i in range(20):
        print(f"  {i:3d} -> {pids[i]}")

    # Save mapping for inspection
    np.savetxt(
        "task_pid_map.csv",
        np.column_stack([task_ids, pids, outputs]),
        delimiter=",",
        header="task_id,pid,avg_u",
        comments="",
    )
    print("\nSaved task_pid_map.csv (task_id,pid,avg_u)")

    # ---- Original outputs summary + plot
    print("\nFirst 10 outputs:", outputs[:10])
    print("Mean output:", outputs.mean())

    plt.figure()
    plt.plot(np.arange(len(outputs)), outputs, marker="o", linestyle="-")
    plt.xlabel("Task index")
    plt.ylabel("Average of solution field, avg(u)")
    plt.title(f"FEniCS parameter sweep: 200 solves ({num_procs} processes)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("parallel_test.png")

    # Optional: a quick “PID assignment” plot to visually confirm distribution
    plt.figure()
    plt.scatter(np.arange(len(pids)), pids)
    plt.xlabel("Task index")
    plt.ylabel("Worker PID")
    plt.title("Which process handled each task?")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("pid_assignment.png")

    plt.show()


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    main(num_procs=4)
