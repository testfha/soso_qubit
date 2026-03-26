import numpy as np
import matplotlib.pyplot as plt
from qutip import Qobj

from scripts.sim.sim import PulseSimulator
from scripts.sim.units import Hz_to_rad, MHz


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------
def normalize(psi: Qobj) -> Qobj:
    return psi.unit()


def logical_basis_from_idle(sim_idle: PulseSimulator):
    evals, evecs = sim_idle.static_hamiltonian.eigenstates()
    psi0L = normalize(evecs[1])
    psi1L = normalize(evecs[2])
    return psi0L, psi1L, evals


def logical_projector_matrix(psi0L: Qobj, psi1L: Qobj) -> np.ndarray:
    return np.column_stack([
        psi0L.full().ravel(),
        psi1L.full().ravel(),
    ])


def six_states():
    return [
        np.array([1, 0], dtype=complex),
        np.array([0, 1], dtype=complex),
        np.array([1, 1], dtype=complex) / np.sqrt(2),
        np.array([1, -1], dtype=complex) / np.sqrt(2),
        np.array([1, 1j], dtype=complex) / np.sqrt(2),
        np.array([1, -1j], dtype=complex) / np.sqrt(2),
    ]


# ------------------------------------------------------------
# system parameters
# ------------------------------------------------------------
b_mhz = np.array([52.5, 74.0, 46.5], dtype=float)
g1, g2, g3 = b_mhz
b = b_mhz * MHz * Hz_to_rad

theta12 = 0.0
theta23 = 0.0
theta13 = 0.0

J12_idle_mhz = 2.0 * (g1 - g3) * (g2 - g3) / (g1 + g2 - 2.0 * g3)
J23_idle_mhz = 0.0
J13_idle_mhz = 0.0

J12_idle = J12_idle_mhz * MHz * Hz_to_rad
J23_idle = J23_idle_mhz * MHz * Hz_to_rad
J13_idle = J13_idle_mhz * MHz * Hz_to_rad

tau = 50e-9  # 50 ns

# target gate
Utarget_Z90 = np.array([
    [np.exp(1j * np.pi / 4), 0],
    [0, np.exp(-1j * np.pi / 4)],
], dtype=complex)

Utarget_Zm90 = np.array([
    [np.exp(-1j * np.pi / 4), 0],
    [0, np.exp(1j * np.pi / 4)],
], dtype=complex)

Utarget = Utarget_Z90


# ------------------------------------------------------------
# 0.5% charge noise
# ------------------------------------------------------------
sigma_noise = 5e-3   # 0.5%
n_noise_avg = 20
rng = np.random.default_rng(1234)


# ------------------------------------------------------------
# idle logical basis
# ------------------------------------------------------------
sim_idle = PulseSimulator(j12=J12_idle, j23=J23_idle, j13=J13_idle, b=b)
sim_idle.set_thetas(theta12, theta23, theta13)

psi0L, psi1L, _ = logical_basis_from_idle(sim_idle)
V = logical_projector_matrix(psi0L, psi1L)

assert np.allclose(V.conj().T @ V, np.eye(2), atol=1e-10)

states6 = six_states()


# ------------------------------------------------------------
# scan grid: final pulse point in (J12, J23)
# ------------------------------------------------------------
J12_scan_mhz = np.linspace(0.0, 10.0, 60)
J23_scan_mhz = np.linspace(0.0, 10.0, 60)

fid_map = np.zeros((len(J12_scan_mhz), len(J23_scan_mhz)))
leak_map = np.zeros((len(J12_scan_mhz), len(J23_scan_mhz)))


# ------------------------------------------------------------
# simulate from idle -> pulse point -> wait 50 ns
# abrupt quench + quasi-static charge noise
# ------------------------------------------------------------
for i, J12_mhz in enumerate(J12_scan_mhz):
    for j, J23_mhz in enumerate(J23_scan_mhz):
        f_noise_list = []
        l_noise_list = []

        noise_samples = rng.normal(0.0, sigma_noise, size=(n_noise_avg, 2))

        for delta12, delta23 in noise_samples:
            J12_noisy_mhz = J12_mhz * (1.0 + delta12)
            J23_noisy_mhz = J23_mhz * (1.0 + delta23)

            sim = PulseSimulator(
                j12=J12_noisy_mhz * MHz * Hz_to_rad,
                j23=J23_noisy_mhz * MHz * Hz_to_rad,
                j13=J13_idle,
                b=b,
            )
            sim.set_thetas(theta12, theta23, theta13)

            f_list = []
            l_list = []

            for psiL in states6:
                psi_target = Utarget @ psiL

                psi_full_init_vec = V @ psiL
                psi_full_init = Qobj(psi_full_init_vec.reshape((-1, 1)), dims=psi0L.dims)

                psi_full_fin = sim.evolve(psi_full_init, np.array([0.0, tau]))[-1]
                psi_full_fin_vec = psi_full_fin.full().ravel()

                psi_log_out = V.conj().T @ psi_full_fin_vec

                f_list.append(abs(np.vdot(psi_target, psi_log_out)) ** 2)
                l_list.append(max(0.0, 1.0 - np.vdot(psi_log_out, psi_log_out).real))

            f_noise_list.append(np.mean(f_list))
            l_noise_list.append(np.mean(l_list))

        fid_map[i, j] = np.mean(f_noise_list)
        leak_map[i, j] = np.mean(l_noise_list)


# ------------------------------------------------------------
# theoretical Z90 point from 1q effective Hamiltonian at tau = 50 ns
# ------------------------------------------------------------
tau_us = tau * 1e6

hz_coeff = ((g1 + g2 - 2.0 * g3) ** 2) / (
    4.0 * (g1**2 + g2**2 - 2.0 * (g1 + g2) * g3 + 2.0 * g3**2)
)

dJ12_th_mhz = 1.0 / (8.0 * hz_coeff * tau_us)
J12_th_mhz = J12_idle_mhz - dJ12_th_mhz   # Z gate: pulse J12 downward
J23_th_mhz = J23_idle_mhz


# ------------------------------------------------------------
# plot infidelity map in log scale
# ------------------------------------------------------------
infid_map = 1.0 - fid_map
infid_map = np.maximum(infid_map, 1e-16)

plt.figure(figsize=(6.5, 5.2))
plt.imshow(
    np.log10(infid_map),
    extent=(J23_scan_mhz[0], J23_scan_mhz[-1], J12_scan_mhz[0], J12_scan_mhz[-1]),
    origin="lower",
    aspect="auto",
)

plt.plot(J23_th_mhz, J12_th_mhz, "ro", markersize=8)

plt.xlabel("J23 (MHz)")
plt.ylabel("J12 (MHz)")
plt.title("infidelity to Z90 after 50 ns from idle, 0.5% charge noise")
plt.colorbar(label="log10(1 - fidelity)")
plt.tight_layout()
plt.show()