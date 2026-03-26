import numpy as np
import matplotlib.pyplot as plt
from qutip import Qobj

from scripts.sim.sim import PulseSimulator
from scripts.sim.units import MHz, Hz_to_rad
# ---------- user-set parameters ----------
b_mhz = np.array([52.5, 74.0, 46.5], dtype=float)
g1, g2, g3 = b_mhz
theta12 = 0.0
theta23 = 0.0
theta13 = 0.0

J12_idle_mhz = 2.0 * (g1 - g3) * (g2 - g3) / (g1 + g2 - 2.0 * g3)
J23_idle_mhz = 0.0
J13_idle_mhz = 0.0

b = b_mhz * MHz * Hz_to_rad
J12_idle = J12_idle_mhz * MHz * Hz_to_rad
J23_idle = J23_idle_mhz * MHz * Hz_to_rad
J13_idle = J13_idle_mhz * MHz * Hz_to_rad


target_gate_name = "Z90"   # or "Z-90"

sigma_noise = 5*1e-3       
n_noise = 200
rng = np.random.default_rng(1234)

t_us = np.arange(0.01, 0.1001, 0.002)
t_s = t_us * 1e-6

# ---------- logical basis at idle ----------
sim_idle = PulseSimulator(j12=J12_idle, j23=J23_idle, j13=J13_idle, b=b)
sim_idle.set_thetas(theta12, theta23, theta13)

evals_idle, evecs_idle = sim_idle.static_hamiltonian.eigenstates()
psi0L = evecs_idle[1].unit()
psi1L = evecs_idle[2].unit()

V = np.column_stack([
    psi0L.full().ravel(),
    psi1L.full().ravel(),
])  # shape (8, 2)

# sanity
assert np.allclose(V.conj().T @ V, np.eye(2), atol=1e-10)

# ---------- six test states ----------
states6 = [
    np.array([1, 0], dtype=complex),
    np.array([0, 1], dtype=complex),
    np.array([1, 1], dtype=complex) / np.sqrt(2),
    np.array([1, -1], dtype=complex) / np.sqrt(2),
    np.array([1, 1j], dtype=complex) / np.sqrt(2),
    np.array([1, -1j], dtype=complex) / np.sqrt(2),
]

# ---------- explicit target gates ----------
Utarget_Z90 = np.array([
    [np.exp(1j * np.pi / 4), 0],
    [0, np.exp(-1j * np.pi / 4)],
], dtype=complex)

Utarget_Zm90 = np.array([
    [np.exp(-1j * np.pi / 4), 0],
    [0, np.exp(1j * np.pi / 4)],
], dtype=complex)

if target_gate_name == "Z90":
    Utarget = Utarget_Z90
elif target_gate_name == "Z-90":
    Utarget = Utarget_Zm90
else:
    raise ValueError("target_gate_name must be 'Z90' or 'Z-90'")

# ---------- analytic amplitude calibration ----------
# effective H_log ≈ hz_coeff * amp * sigma_z
hz_coeff = ((b_mhz[0] + b_mhz[1] - 2 * b_mhz[2]) ** 2) / (
    4 * (b_mhz[0] ** 2 + b_mhz[1] ** 2 - 2 * (b_mhz[0] + b_mhz[1]) * b_mhz[2] + 2 * b_mhz[2] ** 2)
)

def amp_from_time(t_gate_s: float) -> float:
    # amp is positive; actual J12 goes DOWN: J12_gate = J12_idle - amp
    return (np.pi / 4) / (hz_coeff * t_gate_s)

def build_gate_sim(J12_gate: float, J23_gate: float) -> PulseSimulator:
    sim = PulseSimulator(j12=J12_gate, j23=J23_gate, j13=J13_idle, b=b)
    sim.set_thetas(theta12, theta23, theta13)
    return sim

def evolve_full_state(sim: PulseSimulator, psi_full_vec: np.ndarray, t_gate_s: float) -> np.ndarray:
    psi0 = Qobj(psi_full_vec.reshape((-1, 1)), dims=psi0L.dims)
    psi_f = sim.evolve(psi0, np.array([0.0, t_gate_s]))[-1]
    return psi_f.full().ravel()

def fidelity_and_leakage_for_unitary(sim: PulseSimulator, t_gate_s: float):
    fvals = []
    lvals = []

    for psiL in states6:
        psi_target = Utarget @ psiL

        psi_full_in = V @ psiL
        psi_full_out = evolve_full_state(sim, psi_full_in, t_gate_s)
        psi_log_out = V.conj().T @ psi_full_out

        fvals.append(abs(np.vdot(psi_target, psi_log_out)) ** 2)
        lvals.append(1.0 - np.vdot(psi_log_out, psi_log_out).real)

    return float(np.mean(fvals)), float(np.mean(lvals))

def fidelity_and_leakage_noiseless(t_gate_s: float):
    amp = amp_from_time(t_gate_s)
    J12_gate = J12_idle - amp
    J23_gate = J23_idle
    sim = build_gate_sim(J12_gate, J23_gate)
    return fidelity_and_leakage_for_unitary(sim, t_gate_s)

def fidelity_and_leakage_noisy(t_gate_s: float):
    amp = amp_from_time(t_gate_s)
    J12_nom = J12_idle - amp
    J23_nom = J23_idle

    f_noise = []
    l_noise = []

    deltas = rng.normal(0.0, sigma_noise, size=(n_noise, 2))
    for delta12, delta23 in deltas:
        J12_noisy = J12_nom * (1.0 + delta12)
        J23_noisy = J23_nom * (1.0 + delta23)

        sim = build_gate_sim(J12_noisy, J23_noisy)
        f, l = fidelity_and_leakage_for_unitary(sim, t_gate_s)
        f_noise.append(f)
        l_noise.append(l)

    return float(np.mean(f_noise)), float(np.mean(l_noise))

# ---------- scan ----------
infid_noiseless = []
leak_noiseless = []
infid_noisy = []
leak_noisy = []

for tg in t_s:
    f0, l0 = fidelity_and_leakage_noiseless(tg)
    fn, ln = fidelity_and_leakage_noisy(tg)

    infid_noiseless.append(1.0 - f0)
    leak_noiseless.append(l0)
    infid_noisy.append(1.0 - fn)
    leak_noisy.append(ln)

infid_noiseless = np.array(infid_noiseless)
leak_noiseless = np.array(leak_noiseless)
infid_noisy = np.array(infid_noisy)
leak_noisy = np.array(leak_noisy)

# avoid log(0)
eps = 1e-16
infid_noiseless = np.maximum(infid_noiseless, eps)
leak_noiseless = np.maximum(leak_noiseless, eps)
infid_noisy = np.maximum(infid_noisy, eps)
leak_noisy = np.maximum(leak_noisy, eps)

# ---------- plot ----------
plt.figure(figsize=(7, 5))

plt.semilogy(t_us, infid_noiseless, "--", linewidth=2, label="1 - fidelity (noiseless)")
plt.semilogy(t_us, leak_noiseless, "--", linewidth=2, label="leakage (noiseless)")

plt.semilogy(t_us, infid_noisy, "-", linewidth=2, label="1 - fidelity (0.5% noise)")
plt.semilogy(t_us, leak_noisy, "-", linewidth=2, label="leakage (0.5% noise)")

plt.xlabel(r"$t_g$ [$\mu$s]")
plt.ylabel("error")
plt.title(f"{target_gate_name}: infidelity and leakage vs gate time")
plt.legend()
plt.tight_layout()
plt.show()