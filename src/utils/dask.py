from dask.distributed import Client

def create_dask_client(n_workers: int = 4, threads_per_worker: int = 1, memory_limit: str = '4GB') -> Client:
    """Create and return a Dask client for distributed computing."""
    # cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker, memory_limit=memory_limit)
    client = Client(n_workers=n_workers, threads_per_worker=threads_per_worker, memory_limit=memory_limit)
    return client