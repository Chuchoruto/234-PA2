from mpi4py import MPI
import numpy as np

class Communicator(object):
    def __init__(self, comm: MPI.Comm):
        self.comm = comm
        self.total_bytes_transferred = 0

    def Get_size(self):
        return self.comm.Get_size()

    def Get_rank(self):
        return self.comm.Get_rank()

    def Barrier(self):
        return self.comm.Barrier()

    def Allreduce(self, src_array, dest_array, op=MPI.SUM):
        assert src_array.size == dest_array.size
        src_array_byte = src_array.itemsize * src_array.size
        self.total_bytes_transferred += src_array_byte * 2 * (self.comm.Get_size() - 1)
        self.comm.Allreduce(src_array, dest_array, op)

    def Allgather(self, src_array, dest_array):
        src_array_byte = src_array.itemsize * src_array.size
        dest_array_byte = dest_array.itemsize * dest_array.size
        self.total_bytes_transferred += src_array_byte * (self.comm.Get_size() - 1)
        self.total_bytes_transferred += dest_array_byte * (self.comm.Get_size() - 1)
        self.comm.Allgather(src_array, dest_array)

    def Reduce_scatter(self, src_array, dest_array, op=MPI.SUM):
        src_array_byte = src_array.itemsize * src_array.size
        dest_array_byte = dest_array.itemsize * dest_array.size
        self.total_bytes_transferred += src_array_byte * (self.comm.Get_size() - 1)
        self.total_bytes_transferred += dest_array_byte * (self.comm.Get_size() - 1)
        self.comm.Reduce_scatter_block(src_array, dest_array, op)

    def Split(self, key, color):
        return __class__(self.comm.Split(key=key, color=color))

    def Alltoall(self, src_array, dest_array):
        nprocs = self.comm.Get_size()

        # Ensure that the arrays can be evenly partitioned among processes.
        assert src_array.size % nprocs == 0, (
            "src_array size must be divisible by the number of processes"
        )
        assert dest_array.size % nprocs == 0, (
            "dest_array size must be divisible by the number of processes"
        )

        # Calculate the number of bytes in one segment.
        send_seg_bytes = src_array.itemsize * (src_array.size // nprocs)
        recv_seg_bytes = dest_array.itemsize * (dest_array.size // nprocs)

        # Each process sends one segment to every other process (nprocs - 1)
        # and receives one segment from each.
        self.total_bytes_transferred += send_seg_bytes * (nprocs - 1)
        self.total_bytes_transferred += recv_seg_bytes * (nprocs - 1)

        self.comm.Alltoall(src_array, dest_array)

    def myAllreduce(self, src_array, dest_array, op=MPI.SUM):
        """
        A manual implementation of all-reduce using a reduce-to-root
        followed by a broadcast.
        
        Each non-root process sends its data to process 0, which applies the
        reduction operator (by default, summation). Then process 0 sends the
        reduced result back to all processes.
        
        The transfer cost is computed as:
          - For non-root processes: one send and one receive.
          - For the root process: (n-1) receives and (n-1) sends.
        """
        assert src_array.size == dest_array.size, "src and dest arrays must be the same size"
        size = self.Get_size()
        rank = self.Get_rank()
        src_byte = src_array.itemsize * src_array.size
        dest_byte = dest_array.itemsize * dest_array.size

        if rank == 0:
            self.total_bytes_transferred += (size - 1) * (src_byte + dest_byte)
        else:
            self.total_bytes_transferred += src_byte + dest_byte

        self.comm.Reduce(src_array, dest_array, op=op, root=0)

        self.comm.Bcast(dest_array, root=0)

    def myAlltoall(self, src_array, dest_array):
        """
        A manual implementation of all-to-all where each process sends a
        distinct segment of its source array to every other process.
        
        It is assumed that the total length of src_array (and dest_array)
        is evenly divisible by the number of processes.
        
        The algorithm loops over the ranks:
          - For the local segment (when destination == self), a direct copy is done.
          - For all other segments, the process exchanges the corresponding
            portion of its src_array with the other process via Sendrecv.
            
        The total data transferred is updated for each pairwise exchange.
        """
        nprocs = self.Get_size()
        rank = self.Get_rank()
        
        assert src_array.size % nprocs == 0, "src_array size must be divisible by nprocs"
        assert dest_array.size % nprocs == 0, "dest_array size must be divisible by nprocs"
        
        seg_size = src_array.size // nprocs
        seg_bytes = src_array.itemsize * seg_size
        self.total_bytes_transferred += 2 * seg_bytes * (nprocs - 1)
        
        src_array = np.ascontiguousarray(src_array)
        dest_array = np.ascontiguousarray(dest_array)
        
        src_mv = memoryview(src_array)
        dest_mv = memoryview(dest_array)
        
        local_start = rank * seg_size
        local_end = local_start + seg_size
        dest_mv[local_start:local_end] = src_mv[local_start:local_end]
        
        for i in range(1, nprocs):
            send_to = (rank + i) % nprocs
            recv_from = (rank - i + nprocs) % nprocs
            send_start = send_to * seg_size
            send_end = send_start + seg_size
            recv_start = recv_from * seg_size
            recv_end = recv_start + seg_size
            self.comm.Sendrecv(src_mv[send_start:send_end],
                            dest=send_to,
                            recvbuf=dest_mv[recv_start:recv_end],
                            source=recv_from)