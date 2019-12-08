struct PolynomialSystem {
	struct Polynomial *head, *tail;
	int *variables, dimension, degree, size;
};

struct Polynomial {
	int size;
	struct PolyTerm *head, *tail;
	struct Polynomial *next, *prev;
};

struct PolyTerm {
	float coeff;
	struct Monomial *monomial;
	struct PolyTerm *next, *prev;
};

struct Monomial {
	int num_vars;
	struct VarItem **vars;
};

struct VarItem {
	int varNum;
	int varPow;
};

void printPolySystem(struct PolynomialSystem *);
void printTerm(struct PolyTerm *);
void printPoly(struct Polynomial *);
void printMonomial(struct Monomial *);
int grevlex_cmp(struct Monomial *, struct Monomial *);
int grlex_cmp(struct Monomial *, struct Monomial *);
int lex_cmp(struct Monomial *, struct Monomial *);
int totalDegree(struct Monomial *);
