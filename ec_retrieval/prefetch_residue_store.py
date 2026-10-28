"""Bounded residue prefetch with unchanged features and candidate order.

One producer gathers the next batches into two pinned host buffers and uploads
them on a copy stream. CUDA events protect host reuse; record_stream protects
device tensors still used by the consumer's forward/backward. No RNG is used.
The synchronous __getitem__ path is inherited for validation and compatibility.
"""
from __future__ import annotations

from queue import Empty, Full, Queue
from threading import Event, Semaphore, Thread

import numpy as np
import torch

from .residue_store import ResidueFeatureStore


class PrefetchResidueFeatureStore(ResidueFeatureStore):
    def prefetch(self, row_batches, *, depth=2):
        """Return an ordered context-managed iterator over CPU row batches.

        depth bounds queued/in-flight uploads, excluding the current consumer
        batch. Device tensors are owned by each result and never overwritten.
        The caller must use ``with`` so early stops and exceptions join workers.
        """
        if self.device.type != 'cuda':
            raise ValueError('Residue prefetch requires a CUDA mean-feature store')
        if depth not in (1, 2):
            raise ValueError('Prefetch depth must be 1 or 2 to bound memory')
        return _ResiduePrefetch(self, row_batches, depth)


class _ResiduePrefetch:
    def __init__(self, store, row_batches, depth):
        self.store = store
        self.rows = iter(row_batches)
        self.ready = Queue(maxsize=depth)
        self.capacity = Semaphore(depth)
        self.stop = Event()
        self.finished = Event()
        self.error = None
        self.closed = False
        self.host_buffers = [None, None]
        self.host_events = [None, None]
        with torch.cuda.device(store.device):
            self.stream = torch.cuda.Stream(device=store.device)
            # The frozen mean table may have just been uploaded by the caller.
            self.stream.wait_stream(torch.cuda.current_stream(store.device))
        self.worker = Thread(target=self._produce, name='residue-prefetch', daemon=True)
        self.worker.start()

    def _produce(self):
        try:
            with torch.cuda.device(self.store.device), torch.inference_mode(False), torch.no_grad():
                for index, values in enumerate(self.rows):
                    while not self.stop.is_set():
                        if self.capacity.acquire(timeout=.05):
                            break
                    if self.stop.is_set():
                        return
                    if isinstance(values, torch.Tensor):
                        if values.device.type != 'cpu':
                            raise ValueError('Prefetch row metadata must reside on CPU')
                        values = values.detach().numpy()
                    rows = np.array(values, dtype=np.int64, copy=True)
                    if rows.ndim != 1 or not len(rows) or rows.min() < 0 or rows.max() >= self.store.shape[0]:
                        raise ValueError('Nonempty valid one-dimensional candidate rows required')
                    starts = self.store.starts[rows]
                    lengths = self.store.lengths[rows]
                    pointer = np.r_[0, np.cumsum(lengths)]
                    count = int(pointer[-1])
                    positions = np.arange(count, dtype=np.int64)+np.repeat(starts-pointer[:-1], lengths)
                    slot = index % 2
                    if self.host_events[slot] is not None:
                        self.host_events[slot].synchronize()
                    host = self.host_buffers[slot]
                    if host is None or host.shape[0] < count:
                        host = torch.empty((count, self.store.shape[1]), dtype=torch.float16, pin_memory=True)
                        self.host_buffers[slot] = host
                    torch.index_select(self.store.values, 0, torch.from_numpy(positions), out=host[:count])
                    if self.stop.is_set():
                        return
                    with torch.cuda.stream(self.stream):
                        packed = host[:count].to(self.store.device, non_blocking=True)
                        mean = self.store.mean[rows.tolist()]
                        event = torch.cuda.Event()
                        event.record(self.stream)
                    self.host_events[slot] = event
                    payload = ({'mean':mean, 'values':packed, 'offsets_cpu':pointer.tolist()}, event)
                    while not self.stop.is_set():
                        try:
                            self.ready.put(payload, timeout=.05)
                            break
                        except Full:
                            pass
                    if self.stop.is_set():
                        return
        except BaseException as error:
            self.error = error
        finally:
            self.finished.set()

    def __iter__(self):
        return self

    def __next__(self):
        if self.closed:
            raise StopIteration
        while True:
            try:
                features, event = self.ready.get(timeout=.05)
                break
            except Empty:
                if self.finished.is_set():
                    error = self.error
                    self.close()
                    if error is not None:
                        raise RuntimeError('Residue prefetch worker failed') from error
                    raise StopIteration
        stream = torch.cuda.current_stream(self.store.device)
        stream.wait_event(event)
        for key in ('mean', 'values'):
            features[key].record_stream(stream)
        self.capacity.release()
        return features

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.stop.set()
        self.worker.join()
        # Pinned buffers must outlive every asynchronous upload using them.
        self.stream.synchronize()
        while True:
            try:
                self.ready.get_nowait()
            except Empty:
                break
        self.host_buffers = [None, None]
        self.host_events = [None, None]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
