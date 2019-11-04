struct Polynomial * parsePoly(char *);
void printPoly(struct Polynomial *);
void sortPoly(struct Polynomial *);
struct PolyTerm * findLargestTerm(struct Polynomial *, int);

struct PolynomialSystem {
	struct Polynomial* head;
	struct Polynomial* tail;
};

struct Polynomial {
	struct PolyTerm* head;
	struct PolyTerm* tail;
	struct Polynomial* next;
  struct Polynomial* prev;
};

struct PolyTerm {
	float coeff;
	struct VarItem* head;
	struct VarItem* tail;
	struct PolyTerm* next;
  struct PolyTerm* prev;
};

struct VarItem {
	int varNum;
	int varPow;
	struct VarItem* next;
  struct VarItem* prev;
};
