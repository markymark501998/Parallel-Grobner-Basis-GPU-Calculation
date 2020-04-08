# F4/5 SageMath Grobner Basis Algorithm with GPU

This code allows for the F4/5 algorithm written by Martin Albrecht and John Perry for SageMath to be executed with GPU acceleration for the linear algebra operations. Albrecht's source code was modified to allow for the static CUDA shared object library to be executed within Sage. It has been tested for finite field sizes of 32003 & 65521.

Our implementation is able to utilize CUDA because the field sizes are small enough for a double floating point number to represent the full integer with 100% accuracy.

### Requirements:
-SageMath 9.0 (compiled with Python 2.7)<br/>
    -m4<br/>
    -gfortran<br/>
    (m4/gfortran needed to compile Sage)

-CUDA Toolkit (ours was version 10.0 with display driver version 431.XXX)<br/>
-The want to give yourself a headache figuring this out on your own

### Execution:
To compile the shared object (big thanks to Robert T. McGibbon's source code for this):

`$ sudo python setup.py install`

Robert T. McGibbon's Repository: [https://github.com/rmcgibbo/npcuda-example](https://github.com/rmcgibbo/npcuda-example)

Copy the shared object `build/lib.linux-x86_64-2.7/gpuadder.so` to directory where the `f5_2.py` script resides.

To run:

`$ sage test.sage` OR `./sage test.sage` (For machines without a link in `/usr/local/bin/` pointing to sage's executable)

The `test.sage` file contains the Sage code required to execute a simple test run. `f5_2.py` is where the algorithm is located. The source code for the GPU guassian elimination algorithm is in the `src` directory. An example run in `test.sage` file:

`execfile('f5_2.py');`<br/>
`P = PolynomialRing(GF(32003),4,'x');`<br/>
`print(P);`<br/>
`I = sage.rings.ideal.Cyclic(P).homogenize();`<br/>
`print('=======================================================================')`<br/>
`gb = f5_2(I); f5_2.print_stats();`

The above code sets up a polynomial ring and an ideal of finite size (32003 or 65521). Then the Grobner Basis is generated from the ideal.