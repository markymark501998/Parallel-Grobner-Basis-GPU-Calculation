CC=nvcc
TARGET=elim

all:
	$(CC) main.c cublasOps.c textMatrixParsing.c -lcublas -o $(TARGET)

clean:
	rm $(TARGET)
