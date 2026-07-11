"""Build Fable-QFT.ipynb in the Classiq Library glued_trees.ipynb style.

Reproduces Section IV.1 of arXiv:2401.04496 (Figure 2) with a compact notebook
flow: intro -> imports -> algorithm -> helpers -> run -> plot -> references.
"""

import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent
SRC_PATH = ROOT / "Fable-QFT-original.ipynb"
if not SRC_PATH.exists():
    SRC_PATH = ROOT / "Fable-QFT.ipynb"
SRC = json.loads(SRC_PATH.read_text(encoding="utf-8"))


def src(idx: int) -> str:
    return "".join(SRC["cells"][idx]["source"])


nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.11"},
}

cells = []


def md(source: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(source))


def code(source: str) -> None:
    cells.append(nbf.v4.new_code_cell(source))


# ---------------------------------------------------------------------------
# Title (glued_trees-style opening)
# ---------------------------------------------------------------------------
md(
    """# Light-Front QFT Simulation on a Gate-Based Quantum Computer

Consider a (1+1)-dimensional Yukawa model — a fermion and antifermion field coupled to a bosonic field — formulated on the **light front**. Truncating each momentum mode to a small Fock space yields a finite Hamiltonian that can be written as a sum of Pauli strings on 12 qubits. Starting from a single fermion in mode 2, $|f\\rangle_2$, the system evolves under this Hamiltonian and can transition into the state $|f\\rangle_1 \\otimes |\\phi\\rangle_1$, a fermion in mode 1 together with one pion in mode 1.

Classically, the exact evolution $|\\psi(t)\\rangle = e^{-iHt}|\\psi(0)\\rangle$ is tractable here (Hilbert-space dimension $2^{12} = 4096$), but the same Pauli decomposition is exactly what a gate-based quantum computer needs for **Hamiltonian simulation**. This notebook follows Section IV.1 of Vinod & Shaji [[1](#VinodShaji)]: we build the light-front Hamiltonian $H = H_M + H_V + H_S + H_F$, compare exact evolution with a first-order Suzuki–Trotter circuit synthesized through Classiq, and plot the population of $|f\\rangle_1 \\otimes |\\phi\\rangle_1$ as a function of time — reproducing **Figure 2** of the paper."""
)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
code(
    """import itertools
import math
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply

import classiq
from classiq import (
    CReal,
    IndexedPauli,
    Output,
    Pauli,
    QArray,
    QBit,
    SparsePauliOp,
    SparsePauliTerm,
    X,
    allocate,
    create_model,
    qfunc,
    show,
    suzuki_trotter,
    synthesize,
    write_qmod,
)
from classiq.execution import (
    ClassiqBackendPreferences,
    ClassiqSimulatorBackendNames,
    ExecutionPreferences,
    ExecutionSession,
)

RNG_SEED = 1234
np.random.seed(RNG_SEED)

print(f"classiq     {classiq.__version__}")
print(f"numpy       {np.__version__}")
print(f"scipy       {__import__('scipy').__version__}")
print(f"matplotlib  {__import__('matplotlib').__version__}")"""
)

# ---------------------------------------------------------------------------
# Quantum algorithm
# ---------------------------------------------------------------------------
md(
    """## Quantum Algorithm

The light-front Hamiltonian of Section IV.1 decomposes into four terms at successive orders in the coupling $g = \\lambda / \\sqrt{4\\pi}$:

* **$H_M$** — renormalized mass terms (number operators for fermions, antifermions, and bosons);
* **$H_V$** — $O(g)$ vertex interactions (emission/absorption and pair creation);
* **$H_S$** — $O(g^2)$ seagull (instantaneous two-boson exchange);
* **$H_F$** — $O(g^2)$ fork (one particle splitting into three, and reverse).

Fermionic modes use Jordan–Wigner encoding with $Z$-strings; bosonic modes use a compact binary encoding of occupancies $0\\ldots 3$. Each term is built as a dictionary of 12-character Pauli strings, then converted to a sparse matrix for the classical reference evolution.

For the gate-based simulation we approximate

$$U(t) = e^{-iHt} \\approx \\left[\\prod_j e^{-i h_j P_j\\, t / n_T}\\right]^{n_T}$$

using Classiq's `suzuki_trotter` with $n_T = 10$ first-order steps (the paper's Section IV.1 setting). The evolution time $t$ is a classical execution parameter, so the circuit is synthesized once and sampled at many times in a single `ExecutionSession`.

We track the population of the target Fock state $|100\\,000\\,01\\,00\\,00\\rangle$ starting from $|010\\,000\\,00\\,00\\,00\\rangle$."""
)

# ---------------------------------------------------------------------------
# Hamiltonian construction (helpers)
# ---------------------------------------------------------------------------
code(
    "\n".join(
        [
            src(5),
            src(10),
            src(11),
            src(15),
            src(17),
            src(19),
        ]
    )
)

# ---------------------------------------------------------------------------
# Exact reference
# ---------------------------------------------------------------------------
md(
    """## Exact Classical Reference

The exact evolution uses SciPy's `expm_multiply` on the sparse Hamiltonian. Section IV.1 uses a uniform time grid $t \\in [0, 1]$ with step $0.01$. Because the initial and target states both carry the same conserved charges $K = 2$ and $Q = 1$, the dynamics are well approximated by a two-level Rabi oscillation between $|f\\rangle_2$ and $|f\\rangle_1 \\otimes |\\phi\\rangle_1$ — a useful cross-check on the matrix elements extracted from $H$."""
)

code(
    "\n".join(
        [
            src(21),
            src(23),
            """
def style_paper_time_axes(ax, ylabel):
    \"\"\"Axis styling shared with arXiv:2401.04496 Figs. 2-3.\"\"\"
    ax.set_xlim(0, T_MAX)
    ax.set_xticks(np.arange(0, T_MAX + 1e-9, 0.2))
    ax.tick_params(direction="in", top=True, right=True)
    ax.set_xlabel(r"evolution time (in $m_{\\pi}^{-1}$)", fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)


PAPER_WINDOW = "#b8e0b8"  # light green band used in the paper's Fig. 3
""".strip(),
        ]
    )
)

# ---------------------------------------------------------------------------
# Main execution function (glued_trees run_range pattern)
# ---------------------------------------------------------------------------
md(
    """We are now ready to run our main execution function, `run_qft_evolution`. This function synthesizes a single parametric circuit that performs Hamiltonian simulation $e^{-iHt}$ using `suzuki_trotter`. The evolution time `t` is declared as a classical execution parameter (`CReal`), so the circuit is synthesized only once and then sampled at many time values in a single `ExecutionSession`.

The initial Fock state $|f\\rangle_2$ is prepared by flipping the occupied qubits of $|010\\,000\\,00\\,00\\,00\\rangle$. We sample with 8192 shots per time point (matching the paper) and read out the population of $|f\\rangle_1 \\otimes |\\phi\\rangle_1$.

> **Bit-ordering caveat.** Paper kets use **qubit 0 as the leftmost character**; Classiq measurement keys use **qubit 0 as the rightmost character**. The helper `population_from_counts` reverses the paper-notation bitstring before lookup."""
)

code(
    "\n".join(
        [
            src(29),
            src(30),
            """
N_TROTTER = 10
TROTTER_ORDER = 1
NUM_SHOTS = 8192
TIME_STEP_SWEEP = 0.025

execution_preferences = ExecutionPreferences(
    num_shots=NUM_SHOTS,
    random_seed=RNG_SEED,
    backend_preferences=ClassiqBackendPreferences(
        backend_name=ClassiqSimulatorBackendNames.SIMULATOR_STATEVECTOR
    ),
)


def population_from_counts(counts, bitstring):
    \"\"\"Measured population of a paper-notation basis state.\"\"\"
    key = bitstring.replace(" ", "")[::-1]
    return counts.get(key, 0) / sum(counts.values())


def run_time_sweep(quantum_program, time_values):
    with ExecutionSession(quantum_program, execution_preferences) as session:
        results = session.sample([{"t": float(t)} for t in time_values])
    return [result.counts for result in results]


def run_qft_evolution(time_step=TIME_STEP_SWEEP, t_max=T_MAX, export_qmod=True):
    \"\"\"Synthesize the Trotter circuit, optionally export .qmod, and sample a time sweep.\"\"\"
    qprog = synthesize(model)
    show(qprog)

    transpiled = qprog.transpiled_circuit
    gate_counts = dict(transpiled.count_ops)
    circuit_depth = transpiled.depth
    print(f"circuit width : {qprog.data.width} qubits")
    print(f"circuit depth : {circuit_depth}")
    print(f"gate counts   : {gate_counts}")

    if export_qmod:
        write_qmod(model, "qft_simulation", decimal_precision=4)
        import os

        size_mb = os.path.getsize("qft_simulation.qmod") / 1e6
        print(f"qft_simulation.qmod written ({size_mb:.2f} MB)")

    times = np.round(np.arange(0.0, t_max + 1e-9, time_step), 6)
    counts_sweep = run_time_sweep(qprog, times)
    populations = np.array(
        [population_from_counts(c, TARGET_STATE) for c in counts_sweep]
    )
    return qprog, times, populations, counts_sweep, gate_counts, circuit_depth
""".strip(),
        ]
    )
)

# ---------------------------------------------------------------------------
# Run example
# ---------------------------------------------------------------------------
md(
    """The following cell runs `run_qft_evolution` with the Section IV.1 parameters ($n_T = 10$, $\\Delta t = 0.025$ on $[0, 1]$). Synthesis of the 2257-term Hamiltonian takes a couple of minutes."""
)

code(
    """(
    qprog,
    times_trotter,
    populations_trotter,
    counts_sweep,
    gate_counts,
    circuit_depth,
) = run_qft_evolution()

survival_trotter = np.array(
    [population_from_counts(c, INITIAL_STATE) for c in counts_sweep]
)
leak_trotter = 1.0 - populations_trotter - survival_trotter

print(f"{len(times_trotter)} time points sampled, {NUM_SHOTS} shots each")
print(
    f"P_target(t=0.2) = "
    f"{populations_trotter[np.argmin(np.abs(times_trotter - 0.2))]:.4f}"
)
print(f"max leakage out of the 2D subspace (Trotter): {leak_trotter.max():.4f}")"""
)

# ---------------------------------------------------------------------------
# Graph results (glued_trees graph_results pattern)
# ---------------------------------------------------------------------------
md(
    """## Graph Results

The function below plots the exact reference curve together with the Classiq Trotter samples, reproducing Figure 2 of the paper. Error bars show $\\sqrt{p(1-p)/N_{\\mathrm{shots}}}$ shot noise."""
)

code(
    """def graph_figure2(
    times_exact,
    populations_exact,
    times_trotter,
    populations_trotter,
    num_shots=NUM_SHOTS,
    n_trotter=N_TROTTER,
):
    shot_noise = np.sqrt(populations_trotter * (1 - populations_trotter) / num_shots)

    plt.rcParams.update({"font.size": 10, "axes.grid": False})
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.axvspan(
        0,
        0.2,
        color=PAPER_WINDOW,
        alpha=0.95,
        zorder=0,
        label=r"validated window ($t \leq 0.2$, $n_T = 10$)",
    )
    ax.plot(
        times_exact,
        populations_exact,
        ":",
        color="C0",
        lw=1.3,
        zorder=2,
        label="exact evolution $e^{-iHt}$",
    )
    ax.plot(
        times_exact,
        populations_exact,
        "o",
        color="k",
        ms=2.8,
        markeredgewidth=0,
        zorder=3,
    )
    ax.errorbar(
        times_trotter,
        populations_trotter,
        yerr=shot_noise,
        fmt="o",
        ms=4.0,
        color="tab:red",
        ecolor="tab:red",
        capsize=2.0,
        lw=0.8,
        zorder=4,
        label=f"Classiq Suzuki-Trotter ($n_T$={n_trotter}, {num_shots} shots)",
    )
    style_paper_time_axes(
        ax,
        r"probability of the $|f\rangle_1 \otimes |\phi\rangle_1$ state",
    )
    ax.set_ylim(0, 0.25)
    ax.set_yticks(np.arange(0, 0.21, 0.05))
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    plt.show()"""
)

md("""Plot Figure 2: exact evolution vs. Classiq Trotter samples.""")

code(
    """graph_figure2(
    times_exact,
    populations_exact,
    times_trotter,
    populations_trotter,
)"""
)

# ---------------------------------------------------------------------------
# Resource summary + closing notes
# ---------------------------------------------------------------------------
md(
    """## Circuit Resources

With 2257 Pauli terms and $n_T = 10$ Trotter steps, the transpiled circuit contains on the order of $10^4$ gates. The exported `qft_simulation.qmod` file can be inspected in the Classiq IDE or submitted to hardware backends once error mitigation and further Trotter refinement are applied."""
)

code(
    """n_terms = len(hamiltonian_op.terms)
cx_count = gate_counts.get("cx", 0)
total_gates = sum(gate_counts.values())

print(f"logical qubits                : {qprog.data.width}")
print(f"Pauli terms in H              : {n_terms}")
print(f"Trotter repetitions           : {N_TROTTER}")
print(f"term-exponentials in circuit  : {n_terms * N_TROTTER}")
print(f"transpiled depth              : {circuit_depth}")
print(f"total gates                   : {total_gates}")
print(f"two-qubit (CX) gates          : {cx_count}")
print(f"CX per term-exponential       : {cx_count / (n_terms * N_TROTTER):.1f}")"""
)

# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------
md(
    """## References

<a id="VinodShaji"></a>
[1] Gayathree M. Vinod and Anil Shaji, *Simulating Quantum Field Theories on Gate-Based Quantum Computers*, IEEE Transactions on Quantum Engineering **5**, 2500615 (2024). [arXiv:2401.04496](https://arxiv.org/abs/2401.04496), [DOI:10.1109/TQE.2024.3385372](https://doi.org/10.1109/TQE.2024.3385372).

[2] [Classiq Library issue #580](https://github.com/Classiq/classiq-library/issues/580) — contribution tracking this notebook."""
)

nb.cells = cells
out = ROOT / "Fable-QFT.ipynb"
nbf.write(nb, out)
print(f"Wrote {out} ({len(cells)} cells)")
