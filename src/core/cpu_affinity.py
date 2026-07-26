"""NUMA-aware CPU affinity + threading helpers for parallel dataset generation.

The OpenCompMech generators run many independent, single-threaded sparse-FEA
optimizations in parallel (one per pool worker). The solve is a sparse-direct
factorization that is memory-bandwidth bound, so on a multi-NUMA host the right
policy is:

  * ONE worker per PHYSICAL core. Sparse-direct solves saturate a core's
    load/store units and caches; the two SMT threads on a core share those, so
    running 2 workers per core (SMT) contends rather than scales. On the target
    EPYC 7452 (NPS4: 4 NUMA nodes x 8 physical cores) the logical CPUs map as
    CPU N and CPU N+32 -> same physical core, i.e. physical cores = CPUs 0-31.

  * PIN each worker to a fixed core. With Linux first-touch allocation, the
    arrays a worker first writes then live on that core's local NUMA node, so
    each worker's bandwidth stays on its local 2-channel node (~38 GB/s) instead
    of contending across nodes for the ~154 GB/s aggregate.

  * SINGLE-THREADED BLAS inside each worker -- the parallelism is the processes,
    not nested threads (nested threads would oversubscribe and thrash caches).

Usage (ProcessPoolExecutor):

    from multiprocessing import Manager
    from src.core.cpu_affinity import make_affinity_initializer, physical_core_count

    mgr = Manager()
    init, initargs = make_affinity_initializer(mgr, physical_core_count())
    with ProcessPoolExecutor(max_workers=W, initializer=init, initargs=initargs) as ex:
        ...

Manager proxies are used for the shared counter/lock so this works under the
forkserver start method (Python 3.14 default on Linux), where plain
multiprocessing.Value/Lock cannot be passed through initargs.
"""
import os

_BLAS_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def set_single_threaded_blas():
    """Force all BLAS/threadpools to one thread per process."""
    for var in _BLAS_VARS:
        os.environ[var] = "1"


def physical_core_count():
    """Count physical cores via (physical id, core id) pairs in /proc/cpuinfo.

    Falls back to cpu_count // 2 (assumes 2-way SMT) if that can't be parsed.
    """
    try:
        cores = set()
        phys_id = core_id = None
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("physical id"):
                    phys_id = line.split(":", 1)[1].strip()
                elif line.startswith("core id"):
                    core_id = line.split(":", 1)[1].strip()
                elif line.strip() == "":
                    if phys_id is not None and core_id is not None:
                        cores.add((phys_id, core_id))
                    phys_id = core_id = None
        if cores:
            return len(cores)
    except OSError:
        pass
    n = os.cpu_count() or 1
    return max(1, n // 2)


def _affinity_worker_init(counter, lock, n_physical):
    """Pool initializer: single-thread BLAS, then pin to one physical core.

    Each worker atomically claims the next index and pins to logical CPU
    (index % n_physical). On the EPYC 7452 those are CPUs 0-31 -> the first SMT
    thread of each physical core, spread 8-per-node across the 4 NUMA nodes.
    """
    set_single_threaded_blas()
    try:
        with lock:
            idx = counter.value
            counter.value = idx + 1
        core = idx % max(1, n_physical)
        os.sched_setaffinity(0, {core})
    except (AttributeError, OSError):
        # Affinity unsupported (non-Linux, restricted cgroup, ...) -> run
        # unpinned but still single-threaded; correctness is unaffected.
        pass


def make_affinity_initializer(manager, n_physical=None):
    """Build (initializer, initargs) for ProcessPoolExecutor.

    `manager` is a multiprocessing.Manager() (its proxies survive forkserver).
    """
    if n_physical is None:
        n_physical = physical_core_count()
    counter = manager.Value("i", 0)
    lock = manager.Lock()
    return _affinity_worker_init, (counter, lock, n_physical)
