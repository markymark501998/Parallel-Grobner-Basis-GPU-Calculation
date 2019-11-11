CC=nvcc
CC2=gcc
TARGET=elim
TARGET_UTILITY=matrixUtil

all:
	$(CC) main.c cublasOps.cu textMatrixParsing.c -lcublas -o $(TARGET)
	$(CC2) util.c -o $(TARGET_UTILITY)

clean:
	rm $(TARGET)
	rm $(TARGET_UTILITY)
