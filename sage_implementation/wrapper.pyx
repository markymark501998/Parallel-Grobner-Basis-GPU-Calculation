import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "src/manager.hh":
    cdef cppclass C_GPUAdder "GPUAdder":
        C_GPUAdder(np.int32_t*, int)
        void increment()
        void retreive()
        void retreive_to(np.int32_t*, int)
        void F4_5_Elimination(double *, int, int)

    cdef cppclass C_GPUCublas "GPUCublas":
        C_GPUCublas()
        void F4_5_Elimination(double *, int, int)
        void F4_5_Elimination_Finite(double *, int, int, int)

cdef class GPUAdder:
    cdef C_GPUAdder* g
    cdef int dim1

    def __cinit__(self, np.ndarray[ndim=1, dtype=np.int32_t] arr):
        self.dim1 = len(arr)
        self.g = new C_GPUAdder(&arr[0], self.dim1)

    def increment(self):
        self.g.increment()

    def retreive_inplace(self):
        self.g.retreive()

    def retreive(self):
        cdef np.ndarray[ndim=1, dtype=np.int32_t] a = np.zeros(self.dim1, dtype=np.int32)

        self.g.retreive_to(&a[0], self.dim1)

        return a

cdef class GPUCublas:
    cdef C_GPUCublas* g

    def __cinit__(self):
        self.g = new C_GPUCublas()

    def call_cublas_gpu(self, list, rows, cols):
        cdef double* matrix = <double *>malloc(rows * cols * sizeof(double))
        if not matrix:
            raise MemoryError()
        
        for i in range((rows*cols)):
            matrix[i] = float(list[i])

        self.g.F4_5_Elimination(matrix, rows, cols)

        for i in range((rows*cols)):
            list[i] = float(matrix[i])

    def call_cublas_gpu_finite(self, list, rows, cols, field_size):
        cdef double* matrix_finite = <double *>malloc(rows * cols * sizeof(double))
        if not matrix_finite:
            raise MemoryError()
        
        for i in range((rows*cols)):
            matrix_finite[i] = float(list[i])
    
        self.g.F4_5_Elimination_Finite(matrix_finite, rows, cols, field_size)

        for i in range((rows*cols)):
            list[i] = float(matrix_finite[i])