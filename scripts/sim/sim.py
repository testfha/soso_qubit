from collections import namedtuple
from typing import Iterable
from numbers import Real

import matplotlib.pyplot as plt
import numpy as np
from qutip import Qobj, tensor, sigmax, sigmay, sigmaz, qeye

from scripts.sim.pulse import PulseTimeline
from scripts.sim.units import Hz_to_rad, MHz

d: int = 2  # local Hilbert-space dimension (single spin)
N: int = 3  # number of spins

_SX = sigmax()
_SY = sigmay()
_SZ = sigmaz()
_EYE = qeye(d)


def _tensor_operator(pauli: Qobj, position: int) -> Qobj:
    ops = [_EYE] * N
    ops[position] = pauli
    return tensor(ops)


# All single-spin X, Y, Z operators precomputed
SX = tuple(_tensor_operator(_SX, i) for i in range(N))
SY = tuple(_tensor_operator(_SY, i) for i in range(N))
SZ = tuple(_tensor_operator(_SZ, i) for i in range(N))


def _rotation_y(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ])


def anisotropic_exchange(exch: float, theta: float, i: int, j: int) -> Qobj:
    rot_y = _rotation_y(theta)
    s_op = (
            SX[i] * (rot_y[0, 0] * SX[j] + rot_y[0, 1] * SY[j] + rot_y[0, 2] * SZ[j])
            + SY[i] * (rot_y[1, 0] * SX[j] + rot_y[1, 1] * SY[j] + rot_y[1, 2] * SZ[j])
            + SZ[i] * (rot_y[2, 0] * SX[j] + rot_y[2, 1] * SY[j] + rot_y[2, 2] * SZ[j])
    )
    return (exch / 4.0) * s_op


def build_base_ops(
        theta12: float,
        theta23: float,
        theta13: float,
        b: float | Iterable[float] = 1.5,
) -> tuple[Qobj, Qobj, Qobj, Qobj]:
    if not isinstance(b, Iterable):
        b = [b] * N
    b = list(b)

    if len(b) != N:
        raise ValueError(f"b must have length {N}, got {len(b)}")

    h_zeeman = sum(0.5 * b[i] * SZ[i] for i in range(N))

    # Unit exchange operators; later scaled by J_ij(t)
    h_j12_unit = anisotropic_exchange(1.0, theta12, 0, 1)
    h_j23_unit = anisotropic_exchange(1.0, theta23, 1, 2)
    h_j13_unit = anisotropic_exchange(1.0, theta13, 0, 2)

    return h_zeeman, h_j12_unit, h_j23_unit, h_j13_unit


def static_hamiltonian(
        exch12: float,
        exch23: float,
        exch13: float,
        theta12: float,
        theta23: float,
        theta13: float,
        b: float | Iterable[float] = 1.5,
) -> Qobj:
    h_zeeman, h_j12_unit, h_j23_unit, h_j13_unit = build_base_ops(
        theta12, theta23, theta13, b
    )
    return h_zeeman + exch12 * h_j12_unit + exch23 * h_j23_unit + exch13 * h_j13_unit


ExchangeParams = namedtuple("ExchangeParams", ["scale", "exponent", "offset"])


def voltage_to_j(voltage: float, params: ExchangeParams) -> float:
    j_amp = params.scale * np.exp(params.exponent * (voltage - params.offset)) * Hz_to_rad
    return j_amp


class PulseSimulator:
    def __init__(
            self,
            j12: float = 0.0,
            j23: float = 0.0,
            j13: float = 0.0,
            b: float | Iterable[float] = 1.5,
    ):
        self._pulses = PulseTimeline()

        self.theta12 = 0.0
        self.theta23 = 0.0
        self.theta13 = 0.0
        self.b = b

        self.static_j = {
            "J12": j12,
            "J23": j23,
            "J13": j13,
        }

        self._exch_params: dict[str, None | ExchangeParams] = {param: None for param in ("J12", "J23", "J13")}
        self._base_ops_cache = None

    def _base_ops(self) -> tuple[Qobj, Qobj, Qobj, Qobj]:
        if self._base_ops_cache is None:
            self._base_ops_cache = build_base_ops(
                self.theta12,
                self.theta23,
                self.theta13,
                self.b,
            )
        return self._base_ops_cache

    def invalidate_cache(self) -> None:
        self._base_ops_cache = None

    def set_thetas(self, theta12: float, theta23: float, theta13: float) -> None:
        self.theta12 = theta12
        self.theta23 = theta23
        self.theta13 = theta13
        self.invalidate_cache()

    def add_pulse(
            self,
            params: str | Iterable[str],
            amps: float | Iterable[float],
            t_start: float,
            t_end: float,
            relative: bool = True,
    ) -> None:
        if isinstance(params, str):
            params = [params]
        else:
            params = list(params)

        if isinstance(amps, Real):
            amps = [float(amps)] * len(params)
        else:
            amps = list(amps)

        if len(params) != len(amps):
            raise ValueError("Length of params and amps must match")

        for p in params:
            if p not in ("J12", "J23", "J13"):
                raise ValueError(
                    f"Invalid pulse parameter: {p!r}. "
                    f"Must be one of 'J12', 'J23', 'J13'."
                )

        if not relative:
            for p, a in zip(params, amps):
                # if absolute, convert to relative by subtracting the static value
                amps[params.index(p)] = a - self.static_j[p]

        self._pulses.add_pulse(params=params, amps=amps, t_start=t_start, t_end=t_end)

    def clear_pulses(self) -> None:
        self._pulses.clear()

    def set_exchange_params(self, param: str, exch_params: ExchangeParams) -> None:
        if param not in ("J12", "J23", "J13"):
            raise ValueError(
                f"Invalid exchange parameter: {param!r}. "
                f"Must be one of 'J12', 'J23', 'J13'."
            )
        self._exch_params[param] = exch_params

    def get_j_from_voltage(self, param: str, voltage: float) -> float:
        if param not in ("J12", "J23", "J13"):
            raise ValueError(
                f"Invalid exchange parameter: {param!r}. "
                f"Must be one of 'J12', 'J23', 'J13'."
            )
        params = self._exch_params[param]
        if params is None:
            raise ValueError(f"Exchange parameters for {param!r} not set")
        return voltage_to_j(voltage, params)

    def add_voltage_pulse_ramp(
            self,
            param: str,
            amp: float,
            t_start: float,
            t_end: float,
            t_ramp_in: float,
            t_ramp_out: float,
            sampling_rate: float,
            exchange_params: ExchangeParams | None = None,
    ):
        ramp_in_voltage = np.linspace(0.0, amp, int(t_ramp_in * sampling_rate), endpoint=False)
        ramp_out_voltage = np.linspace(amp, 0.0, int(t_ramp_out * sampling_rate), endpoint=False)
        for idx, (v_in, v_out) in enumerate(zip(ramp_in_voltage, ramp_out_voltage)):
            t_start_point = t_start + idx / sampling_rate
            self.add_voltage_pulse(param, v_in, t_start_point, t_end, exchange_params)
            self.add_voltage_pulse(param, v_out, t_end-t_ramp_out, t_end, exchange_params)

        # Add the main pulse with the full amplitude
        self.add_voltage_pulse(param, amp, t_start+t_ramp_in, t_end-t_ramp_out, exchange_params)

    def add_voltage_pulse(
            self,
            param: str,
            amp: float,
            t_start: float,
            t_end: float,
            exchange_params: ExchangeParams | None = None,
    ):
        if exchange_params:
            j_amp = voltage_to_j(amp, exchange_params)
        else:
            j_amp = self.get_j_from_voltage(param, amp)

        self.add_pulse(param, j_amp, t_start, t_end, relative=False)

    def build(self) -> None:
        self._pulses.build()

    def h_of_t(self, t: float) -> Qobj:
        hz, hj12_unit, hj23_unit, hj13_unit = self._base_ops()

        j12 = self.static_j["J12"] + self._pulses.amplitude(t, "J12")
        j23 = self.static_j["J23"] + self._pulses.amplitude(t, "J23")
        j13 = self.static_j["J13"] + self._pulses.amplitude(t, "J13")

        return hz + j12 * hj12_unit + j23 * hj23_unit + j13 * hj13_unit

    def evolve(self, psi0: Qobj, tlist: np.ndarray) -> np.ndarray:
        """
        Exact piecewise-constant evolution for a closed system.

        The Hamiltonian is assumed constant between pulse breakpoints.
        Returns the states at the requested times in tlist.
        """
        self.build()

        tlist = np.asarray(tlist, dtype=float)
        if tlist.ndim != 1 or len(tlist) == 0:
            raise ValueError("tlist must be a nonempty 1D array")
        if np.any(np.diff(tlist) < 0):
            raise ValueError("tlist must be sorted in ascending order")

        hz, hj12_unit, hj23_unit, hj13_unit = self._base_ops()

        def h_at(t: float) -> Qobj:
            j12 = self.static_j["J12"] + self._pulses.amplitude(t, "J12")
            j23 = self.static_j["J23"] + self._pulses.amplitude(t, "J23")
            j13 = self.static_j["J13"] + self._pulses.amplitude(t, "J13")
            return hz + j12 * hj12_unit + j23 * hj23_unit + j13 * hj13_unit

        # Insert switching times so every propagation segment has constant H
        breakpoints = [
            t for t in self._pulses.breakpoints
            if tlist[0] < t < tlist[-1]
        ]
        all_times = np.array(sorted(set(tlist.tolist() + breakpoints)), dtype=float)

        states_at = {}
        psi = psi0
        states_at[float(all_times[0])] = psi

        for t0, t1 in zip(all_times[:-1], all_times[1:]):
            dt = t1 - t0
            if dt == 0.0:
                continue

            # Evaluate in the middle of the interval to avoid discontinuities
            t_mid = 0.5 * (t0 + t1)
            h = h_at(t_mid)
            u = (-1j * h * dt).expm()
            psi = u * psi
            states_at[float(t1)] = psi

        return np.array([states_at[float(t)] for t in tlist], dtype=object)

    @property
    def static_hamiltonian(self) -> Qobj:
        return static_hamiltonian(
            exch12=self.static_j["J12"],
            exch23=self.static_j["J23"],
            exch13=self.static_j["J13"],
            theta12=self.theta12,
            theta23=self.theta23,
            theta13=self.theta13,
            b=self.b,
        )


if __name__ == "__main__":
    b = [
        52.5 * MHz * Hz_to_rad,
        74 * MHz * Hz_to_rad,
        46.5 * MHz * Hz_to_rad,
    ]

    j12 = np.linspace(0 * MHz * Hz_to_rad, 30 * MHz * Hz_to_rad, 41)
    j23 = np.linspace(0 * MHz * Hz_to_rad, 30 * MHz * Hz_to_rad, 41)
    pop = np.zeros((len(j12), len(j23)))

    tlist = [0, 200e-9]

    for i12, exch12 in enumerate(j12):
        for i23, exch23 in enumerate(j23):
            sim = PulseSimulator(b=b)

            sim.add_pulse("J12", exch12, 0.0, 200e-9)
            sim.add_pulse("J23", exch23, 0.0, 200e-9)

            evals, evecs = sim.static_hamiltonian.eigenstates()
            psi0 = evecs[1]

            states = sim.evolve(psi0, tlist)
            psi_f = states[-1]

            pop[i12, i23] = abs(psi0.overlap(psi_f)) ** 2

    plt.figure()
    plt.imshow(
        pop,
        extent=(
            j12[0] / (MHz * Hz_to_rad),
            j12[-1] / (MHz * Hz_to_rad),
            j23[0] / (MHz * Hz_to_rad),
            j23[-1] / (MHz * Hz_to_rad),
        ),
        origin="lower",
        aspect="auto",
    )
    plt.xlabel("J23 (MHz)")
    plt.ylabel("J12 (MHz)")
    plt.title("Return probability after 200 ns")
    plt.colorbar(label="Population")
    plt.show()
