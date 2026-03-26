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
b_mhz = np.array([52.5, 74.0, 46.5], dtype=float)
g1, g2, g3 = b_mhz

b = b_mhz * MHz * Hz_to_rad

theta12 = 0.0
theta23 = 0.0
theta13 = 0.0

# exact idle point from the 1q theory
J12_idle_mhz = 2.0 * (g1 - g3) * (g2 - g3) / (g1 + g2 - 2.0 * g3)
J23_idle_mhz = 0.0
J13_idle_mhz = 0.0

J12_idle = J12_idle_mhz * MHz * Hz_to_rad
J23_idle = J23_idle_mhz * MHz * Hz_to_rad
J13_idle = J13_idle_mhz * MHz * Hz_to_rad

print(f"J12_idle = {J12_idle_mhz:.6f} MHz")

# -----------------------------
# 0.5% quasi-static charge noise
# noise is applied to deviation from idle:
# J12_noisy = J12_idle + (J12_val - J12_idle) * (1 + delta)
# -----------------------------
sigma_noise = 5e-3
n_noise_avg = 50
rng = np.random.default_rng(1234)

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
J12_scan = np.linspace(0.0, 18.0, 81) * MHz * Hz_to_rad

signal = np.zeros((len(J12_scan), len(taus)))

# -----------------------------
# Ramsey-like map with 0.5% charge noise
# -----------------------------
for i, J12_val in enumerate(J12_scan):
    noise_samples = rng.normal(0.0, sigma_noise, size=n_noise_avg)

    for j, tau in enumerate(taus):
        vals = []

        for delta in noise_samples:
            # noise on deviation from idle
            J12_noisy = J12_idle + (J12_val - J12_idle) * (1.0 + delta)

            sim = PulseSimulator(j12=J12_noisy, j23=J23_idle, j13=J13_idle, b=b)
            sim.set_thetas(theta12, theta23, theta13)

            vals.append(ramsey_return_prob(sim, psi_plus, tau))

        signal[i, j] = np.mean(vals)

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(6, 5))
plt.imshow(
    signal,
    extent=(
        taus[0] * 1e6,
        taus[-1] * 1e6,
        J12_scan[0] / (MHz * Hz_to_rad),
        J12_scan[-1] / (MHz * Hz_to_rad),
    ),
    origin="lower",
    aspect="auto",
    cmap="Blues",
)
plt.xlabel(r'$\tau$ [$\mu$s]')
plt.ylabel(r'$J_{12}$ [MHz]')
plt.title('Ramsey-like return probability (0.5% charge noise)')
plt.colorbar(label='P(return to |+>)')
plt.tight_layout()
plt.show()