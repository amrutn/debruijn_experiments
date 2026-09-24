"""
Accuracy vs memory window expressed as a FRACTION of each problem's own reasoning,
quantile-binned from the cached knockout-generation results (no new generation).

This is a thin wrapper: the plotting lives in entropy_exp.py
(plot_accuracy_vs_fraction_rebinned) so it shares the exact axis and styling
helpers of the other Fig. 6 panels. The same plot is also produced by the default
knockout-generation run / --plot-only; this script just regenerates it alone.
See the docstring of rebin_accuracy_by_fraction for the selection-bias caveat.

Usage (from benchmarks/, same env as entropy_exp.py):
    python plot_accuracy_vs_fraction.py                       # 14B solid, 32B dashed
    python plot_accuracy_vs_fraction.py Qwen/Qwen3-14B        # one model only
"""
import sys

if __name__ == "__main__":
    models = sys.argv[1:] or ["Qwen/Qwen3-14B", "Qwen/Qwen3-32B"]
    sys.argv = [sys.argv[0]]            # entropy_exp parses argv on import; give it none
    import entropy_exp as E
    E.plot_accuracy_vs_fraction_rebinned(models)
