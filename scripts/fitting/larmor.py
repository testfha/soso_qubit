import numpy as np

from lmfit import Model as LmFitModel

def freq_rabi(x, amplitude, offset, center_frequency, rabi_frequency, angle):
    detuning = x - center_frequency
    s = rabi_frequency * rabi_frequency + detuning * detuning
    t = angle / rabi_frequency
    sin_value = np.sin(np.sqrt(s) * t / 2)
    return (
        offset + amplitude * sin_value * sin_value * rabi_frequency * rabi_frequency / s
    )

model = LmFitModel(freq_rabi)
