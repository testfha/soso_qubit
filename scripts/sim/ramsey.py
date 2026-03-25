import numpy as np
import matplotlib.pyplot as plt
from qutip import Qobj

from .sim import PulseSimulator, ExchangeParams, MHz, Hz_to_rad

# -----------------------------
# Helpers
# -----------------------------
def normalize(psi: Qobj) -> Qobj:
    return psi.unit()

def logical_basis_from_idle(sim_idle: PulseSimulator):
    """Take the 2 logical states from the idle Hamiltonian."""
    evals, evecs = sim_idle.static_hamiltonian.eigenstates()
    # adjust indices if your logical pair is not [1], [2]
    psi0L = normalize(evecs[1])
    psi1L = normalize(evecs[2])
    return psi0L, psi1L, evals

def plus_state(psi0L: Qobj, psi1L: Qobj) -> Qobj:
    return normalize(psi0L + psi1L)

def ramsey_return_prob(sim: PulseSimulator, psi_init: Qobj, tau: float) -> float:
    """Free evolution for time tau, then project back onto initial superposition."""
    states = sim.evolve(psi_init, np.array([0.0, tau]))
    psi_f = states[-1]
    return abs(psi_init.overlap(psi_f)) ** 2

# -----------------------------
# Device / model parameters
# -----------------------------
# Example Zeeman splittings from their code
b = [
    52.5 * MHz * Hz_to_rad,
    74.0 * MHz * Hz_to_rad,
    46.5 * MHz * Hz_to_rad,
]

# Example SO angles; replace with your actual values
theta12 = 0.0
theta23 = 0.0
theta13 = 0.0

# Idle point in EXCHANGE units.
# Replace by your actual idle J12, J23, J13 if known.
J12_idle = 10* MHz * Hz_to_rad
J23_idle = 0* MHz * Hz_to_rad
J13_idle = 0.0

# -----------------------------
# Build idle simulator and logical basis
# -----------------------------
sim_idle = PulseSimulator(j12=J12_idle, j23=J23_idle, j13=J13_idle, b=b)
sim_idle.set_thetas(theta12, theta23, theta13)

psi0L, psi1L, evals = logical_basis_from_idle(sim_idle)
psi_plus = plus_state(psi0L, psi1L)

# -----------------------------
# Scan axes
# -----------------------------
taus = np.linspace(0.0, 2.0e-6, 201)   # 0 to 2 us
J12_scan = np.linspace(0.0, 18.0, 81) * MHz * Hz_to_rad  # adjust range

signal = np.zeros((len(J12_scan), len(taus)))

# -----------------------------
# Ramsey-like map
# -----------------------------
for i, J12_val in enumerate(J12_scan):
    sim = PulseSimulator(j12=J12_val, j23=J23_idle, j13=J13_idle, b=b)
    sim.set_thetas(theta12, theta23, theta13)

    for j, tau in enumerate(taus):
        signal[i, j] = ramsey_return_prob(sim, psi_plus, tau)

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(6, 5))
plt.imshow(
    signal,
    extent=(taus[0] * 1e6, taus[-1] * 1e6,
            J12_scan[0] / (MHz * Hz_to_rad), J12_scan[-1] / (MHz * Hz_to_rad)),
    aspect='auto',
    cmap='Blues',
)
plt.xlabel(r'$\tau$ [$\mu$s]')
plt.ylabel(r'$J_{12}$ [MHz]')
plt.title('Ramsey-like return probability')
plt.colorbar(label='P(return to |+>)')
plt.tight_layout()
plt.show()