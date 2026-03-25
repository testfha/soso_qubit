from collections import namedtuple, defaultdict
from bisect import bisect_right
import matplotlib.pyplot as plt

Pulse = namedtuple("Pulse", ("params", "amps", "t_start", "t_end"))


class PulseTimeline:
    def __init__(self, pulses=None):
        self.pulses = []
        self._built = False

        self.all_params = []
        self.breakpoints = []
        self.states = []

        if pulses is not None:
            for pulse in pulses:
                self.add_pulse(pulse)

    def add_pulse(self, pulse=None, *, params=None, amps=None, t_start=None, t_end=None):
        """
        Add a pulse.

        Either pass a Pulse instance:
            add_pulse(Pulse(...))

        or keyword arguments:
            add_pulse(params=[...], amps=[...], t_start=..., t_end=...)
        """
        if pulse is None:
            pulse = Pulse(params=params, amps=amps, t_start=t_start, t_end=t_end)

        if len(pulse.params) != len(pulse.amps):
            raise ValueError("pulse.params and pulse.amps must have the same length")
        if pulse.t_end < pulse.t_start:
            raise ValueError("pulse.t_end must be >= pulse.t_start")

        self.pulses.append(pulse)
        self._built = False

    def clear(self):
        """Remove all pulses and reset the timeline."""
        self.pulses.clear()
        self.all_params = []
        self.breakpoints = []
        self.states = []
        self._built = False

    def build(self):
        """Build the internal piecewise-constant representation."""
        events = defaultdict(lambda: defaultdict(float))
        all_params = set()

        for pulse in self.pulses:
            for p, a in zip(pulse.params, pulse.amps):
                all_params.add(p)
                events[pulse.t_start][p] += a
                events[pulse.t_end][p] -= a

        self.all_params = sorted(all_params)
        self.breakpoints = sorted(events.keys())

        current = defaultdict(float)
        self.states = []

        for t in self.breakpoints:
            for p, delta in events[t].items():
                current[p] += delta
                if abs(current[p]) < 1e-15:
                    current[p] = 0.0
            self.states.append(dict(current))

        self._built = True

    def _ensure_built(self):
        if not self._built:
            self.build()

    def __call__(self, t):
        """
        Return amplitudes at time t as a dict {param: amplitude}.

        Convention: pulses are active on [t_start, t_end).
        """
        self._ensure_built()

        if not self.breakpoints:
            return {}

        idx = bisect_right(self.breakpoints, t) - 1
        if idx < 0:
            return {}

        # Only return nonzero entries
        return {p: a for p, a in self.states[idx].items() if abs(a) > 1e-15}

    def amplitude(self, t, param):
        """Return amplitude of a single parameter at time t."""
        return self(t).get(param, 0.0)

    def intervals(self):
        """
        Return a piecewise-constant timeline as:
        [(t0, t1, {param: amp, ...}), ...]
        """
        self._ensure_built()

        out = []
        for i in range(len(self.breakpoints) - 1):
            t0 = self.breakpoints[i]
            t1 = self.breakpoints[i + 1]
            state = {
                p: a for p, a in self.states[i].items()
                if abs(a) > 1e-15
            }
            out.append((t0, t1, state))
        return out

    def plot(self, params=None, t_min=None, t_max=None, ax=None, show=True):
        """
        Plot the timeline as step functions.

        Parameters
        ----------
        params : list[str] or None
            Which parameters to plot. If None, plot all.
        t_min, t_max : float or None
            Time range to display. If omitted, inferred from pulses.
        ax : matplotlib axis or None
            Existing axis to draw on.
        show : bool
            Whether to call plt.show().
        """
        self._ensure_built()

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))

        if not self.pulses:
            ax.set_title("Pulse timeline")
            ax.set_xlabel("t")
            ax.set_ylabel("amplitude")
            if show:
                plt.show()
            return ax

        if params is None:
            params = self.all_params
        else:
            params = list(params)

        pulse_t_min = min(p.t_start for p in self.pulses)
        pulse_t_max = max(p.t_end for p in self.pulses)

        if t_min is None:
            t_min = pulse_t_min
        if t_max is None:
            t_max = pulse_t_max

        if t_max < t_min:
            raise ValueError("t_max must be >= t_min")

        # Collect all relevant x points
        xs = [t_min]
        xs.extend(t for t in self.breakpoints if t_min < t < t_max)
        xs.append(t_max)
        xs = sorted(set(xs))

        for param in params:
            ys = [self.amplitude(t, param) for t in xs[:-1]]
            ys.append(ys[-1] if ys else 0.0)  # for step(..., where='post')
            ax.step(xs, ys, where="post", label=param)

        ax.set_xlim(t_min, t_max)
        ax.set_xlabel("t")
        ax.set_ylabel("amplitude")
        ax.set_title("Pulse timeline")
        ax.legend()
        ax.grid(True, alpha=0.3)

        if show:
            plt.show()

        return ax

if __name__ == "__main__":
    timeline = PulseTimeline()
    timeline.add_pulse(params=["A"], amps=[1.0], t_start=0.0, t_end=1.0)
    timeline.add_pulse(params=["B"], amps=[0.5], t_start=0.5, t_end=1.5)
    timeline.add_pulse(params=["A"], amps=[-1.0], t_start=1.0, t_end=2.0)
    print(timeline.amplitude(0.75, "B"))
    print(timeline.amplitude(0.75, "A"))
    print(timeline.intervals())
    timeline.plot()