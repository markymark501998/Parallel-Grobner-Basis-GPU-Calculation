Guassian Elimination & Matrix Utility Tool
==============================================================================

Compiling & Running Instructions
--------------------------------------------------------
1:  Inside of the "elimination" folder in a terminal, run "make" and execute to compile the code

2:  To create a matrix with which to use in the algorithm, run "./matrixUtil -createMatrix [rows] [cols] [output filename]"
Note: It will fill the matrix with random float values in every matrix position

3:  To execute the main algorithm, run "./elim [input filename] [-dontPrint]<- this last argument is optional>" Note: I recommend running with -dontPrint as larger matrices will result in thousands of console outputs that will overwhelm the user. Only take away this argument with small matrices