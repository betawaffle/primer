"""Training-data generators, one per vocabulary group. See groups.py."""
import os

# Generation runs one process per core; multithreaded BLAS inside each process
# only fights over the same cores (measured: 4x slower). Set before numpy loads.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from .groups import GROUPS, generate  # noqa: E402

__all__ = ["GROUPS", "generate"]
