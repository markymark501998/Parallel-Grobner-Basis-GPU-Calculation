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
	int degree, *exponents;
	struct PolyTerm *next, *prev;

	// deprecated
	struct Monomial *monomial;
};

struct Monomial {
	int num_vars;
	struct VarItem **vars;
};

struct VarItem {
	int varNum;
	int varPow;
};

void printPolySystem2(struct PolynomialSystem *);
void printMonomial2(struct PolyTerm *, int *, int);
int grevlex_cmp(struct Monomial *, struct Monomial *);
int grlex_cmp(struct Monomial *, struct Monomial *);
int lex_cmp(struct Monomial *, struct Monomial *);
int mono_cmp(int *, int *, int, enum MonomialOrdering);
int totalDegree(struct Monomial *);
