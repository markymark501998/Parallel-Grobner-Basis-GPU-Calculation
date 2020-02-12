CC=nvcc
CC2=gcc
TARGET=elim
TARGET_UTILITY=util

all:
	$(CC) main.c cublasOps.cu cublasOps2.cu textMatrixParsing.c -lcublas -lcusparse -o $(TARGET)
	$(CC2) util.c -o $(TARGET_UTILITY)

debug:
	$(CC) -pg main.c cublasOps.cu cublasOps2.cu textMatrixParsing.c -lcublas -lcusparse -o $(TARGET)
	$(CC2) util.c -o $(TARGET_UTILITY)

clean:
	rm $(TARGET)
	rm $(TARGET_UTILITY)
