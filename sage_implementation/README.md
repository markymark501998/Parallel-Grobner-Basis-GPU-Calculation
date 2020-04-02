## F4/5 SageMath Grobner Basis Algorithm

This code allows for the F4/5 algorithm written by Martin Albrecht and John Perry for SageMath to be executed with GPU acceleration. Albrecht's source code was modified to allow for the static CUDA shared object library to be executed within Sage.

To create shared object:

`$ sudo python setup.py install`

to run:

`$ sage test.sage`

"test.sage" contains the Sage code required to execute a simple test run. The contents are as follows:


`execfile('f5_2.py');`
`P = PolynomialRing(GF(32003),4,'x');`
`print(P);`
`I = sage.rings.ideal.Cyclic(P).homogenize();`
`print('=======================================================================')`
`gb = f5_2(I); f5_2.print_stats();`