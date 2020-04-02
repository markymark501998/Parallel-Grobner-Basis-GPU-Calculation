# F4/5 SageMath Grobner Basis Algorithm with GPU

This code allows for the F4/5 algorithm written by Martin Albrecht and John Perry for SageMath to be executed with GPU acceleration for the linear algebra operations. Albrecht's source code was modified to allow for the static CUDA shared object library to be executed within Sage.

### Requirements:
-SageMath 9.0 (compiled with Python 2.7)<br/>
    -m4<br/>
    -gfortran

-CUDA Toolkit (ours was version 10.0 with display driver 431)<br/>
-The want to give yourself a headache figuring this out on your own

To compile the shared object:

`$ sudo python setup.py install`

Copy the shared object `build/lib.linux-x86_64-2.7/gpuadder.so` to this directory.

To run:

`$ sage test.sage` OR `./sage test.sage` (For machines without a link in `/usr/local/bin/` pointing to sage's executable)

The `test.sage` file contains the Sage code required to execute a simple test run. `f5_2.py` is where the algorithm is located. The source code for the GPU guassian elimination algorithm is in the `src` directory. The contents of `test.sage` are as follows:

`execfile('f5_2.py');`<br/>
`P = PolynomialRing(GF(32003),4,'x');`<br/>
`print(P);`<br/>
`I = sage.rings.ideal.Cyclic(P).homogenize();`<br/>
`print('=======================================================================')`<br/>
`gb = f5_2(I); f5_2.print_stats();`

The above code sets up a polynomial ring and an ideal of finite size (32003). Then the Grobner Basis is generated.