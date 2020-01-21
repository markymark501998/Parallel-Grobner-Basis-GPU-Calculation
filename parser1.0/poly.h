enum MonomialOrdering {grevlex, grlex, lex};

struct PolynomialSystem {
	struct Polynomial *head, *tail;
	int *variables, dimension, degree, size;
	enum MonomialOrdering order;
};

struct Polynomial {
	int size;
	struct PolyTerm *head, *tail;
	struct Polynomial *next, *prev;
};

struct PolyTerm {
	float coeff;
	struct PolyTerm *next, *prev;
	struct Monomial *monomial;
};

struct Monomial {
	int *exponents, degree;
};

struct VarItem {
	int varNum;
	int varPow;
};

void printPolySystem2(struct PolynomialSystem *);
void printMonomial2(struct PolyTerm *, int *, int);
void printMonomial(int *, int *, int);
int mono_cmp(int *, int *, int, enum MonomialOrdering);
