import gpuadder

def gpu_test():
    instance = gpuadder.GPUCublas()
    test_list = [0, 9, 5, 0, 0, 0, .553970, 2, 3, 0.156679, 0.798440, 0, 0, 2, 4, 0.197551, 0, 0, 12, 0, 0, 0, 0, 0, 0]
    instance.call_cublas_gpu(test_list, 5, 5)
    print(test_list)