int GuassianEliminationV1 (float**, int, int, int);
int GuassianEliminationV1Rref (float**, int, int, int, int);

int FGL_Algorithm (float**, int, int, int, int, int);
int FGL_Algorithm_Double (double**, int, int, int, int, int);
int FGL_Algorithm_Double_NewScalingMethod (double**, int, int, int, int, int);

int F4_5_GuassianElimination (double **, int, int, int, int);
int F4_5_GuassianEliminationCuSparse (double **, int, int, int, int);
int F4_5_GuassianEliminationCuSparseMHVersion (double **, int, int, int, int);

float** parseInputMatrix(FILE *, int, int*, int*);
double** parseInputMatrixDouble(FILE *, int, int*, int*);

void printMatrix(float **, int, int);

void printMatrixWithLimits(float **, int, int, int);
void printMatrixWithLimitsDouble(double **, int, int, int);

void printCublasMatrixArray(float * , int);
void printCublasMatrixArrayConverted (float*, int, int);

void printStandardIntArray(int*, int);
void printStandardDoubleArray(double*, int);

void printSparseMatrixArray (float**, int, int, int);
void printSparseMatrixArrayConverted (float*, int, int, int);

void printSparseMatrixArrayDouble (double**, int, int, int);
void printSparseMatrixArrayConvertedDouble (double*, int, int, int);