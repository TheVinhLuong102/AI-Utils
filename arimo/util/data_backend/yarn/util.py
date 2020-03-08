__all__ = 'alloc_executors', 'optim_alloc'


NODE_MEM_OVERHEAD_MARGIN_PROPORTION = .025

DRIVER_MEM_MIN_GiB = 1.68
EXECUTOR_MEM_GiB_OPTIM_RANGE = range(24, 33)   # sufficient for Big Data & ML workloads

CONTAINER_MEM_OVERHEAD_MARKUP_PROPORTION = .1
CONTAINER_MEM_OVERHEAD_MARKUP_MIN_GiB = .384

_DRIVER_MEM_MIN_DB_INCL_OVERHEAD = \
    max(CONTAINER_MEM_OVERHEAD_MARKUP_MIN_GiB,
        CONTAINER_MEM_OVERHEAD_MARKUP_PROPORTION * DRIVER_MEM_MIN_GiB)


def alloc_executors(node_mem_gib: int, executor_mem_gib: int) -> dict:   # TODO Py3.8: positional-/keyword-only
    avail_mem_for_all_containers = \
        (1 - NODE_MEM_OVERHEAD_MARGIN_PROPORTION) * node_mem_gib

    avail_mem_for_executors = \
        avail_mem_for_all_containers - _DRIVER_MEM_MIN_DB_INCL_OVERHEAD

    executor_mem_overhead_gib = \
        max(CONTAINER_MEM_OVERHEAD_MARKUP_MIN_GiB,
            CONTAINER_MEM_OVERHEAD_MARKUP_PROPORTION * executor_mem_gib)

    executor_mem_gib_incl_overhead = \
        executor_mem_gib + executor_mem_overhead_gib

    n_executors = \
        int(avail_mem_for_executors /
            executor_mem_gib_incl_overhead)

    avail_for_driver_mem_gib_incl_overhead = \
        avail_mem_for_all_containers - \
        n_executors * executor_mem_gib_incl_overhead

    avail_for_driver_mem_gib = \
        min(avail_for_driver_mem_gib_incl_overhead - CONTAINER_MEM_OVERHEAD_MARKUP_MIN_GiB,
            avail_for_driver_mem_gib_incl_overhead / (1 + CONTAINER_MEM_OVERHEAD_MARKUP_PROPORTION))

    return dict(
            n_executors=n_executors,
            avail_for_driver_mem_gib=avail_for_driver_mem_gib)


def optim_alloc(node_mem_gib: int) -> dict:   # TODO Py3.8: positional-only
    for executor_mem_gib, alloc_details in \
            sorted(
                [(executor_mem_gib, alloc_executors(node_mem_gib, executor_mem_gib))
                 for executor_mem_gib in EXECUTOR_MEM_GiB_OPTIM_RANGE],
                key=lambda option: option[1]['avail_for_driver_mem_gib']):
        avail_for_driver_mem_gib = alloc_details['avail_for_driver_mem_gib']

        if avail_for_driver_mem_gib >= DRIVER_MEM_MIN_GiB:
            return dict(
                    executor_mem_gib=executor_mem_gib,
                    n_executors=alloc_details['n_executors'],
                    avail_for_driver_mem_gib=avail_for_driver_mem_gib)
