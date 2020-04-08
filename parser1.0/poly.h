struct PolynomialSystem {
	struct Polynomial *head;
	struct Polynomial *tail;
	int size;
};

struct Polynomial {
	struct PolyTerm *head;
	struct PolyTerm *tail;
	struct Polynomial *next;
  struct Polynomial *prev;
};

struct PolyTerm {
	float coeff;
	int num_vars;
	struct VarItem **vars;
	struct PolyTerm *next;
  struct PolyTerm *prev;
};

struct VarItem {
	int varNum;
	int varPow;
};

void printPolySystem(struct PolynomialSystem *);
void printTerm(struct PolyTerm *);
void printPoly(struct Polynomial *);
int grevlex_cmp(struct PolyTerm *, struct PolyTerm *);
int grlex_cmp(struct PolyTerm *, struct PolyTerm *);
int lex_cmp(struct PolyTerm *, struct PolyTerm *);
