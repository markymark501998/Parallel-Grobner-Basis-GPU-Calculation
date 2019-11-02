void myFunc(void);
int indexOf(char *, char);
int indexOfStart(char *, char, int);
void substring(char [], char [], int, int);

struct PolynomialSystem {
	struct Polynomial* head;
	struct Polynomial* tail;
};

struct Polynomial {
	struct PolyTerm* head;
	struct PolyTerm* tail;
	struct Polynomial* next;
};

struct PolyTerm {
	float coeff;
	struct VarItem* head;
	struct VarItem* tail;
	struct PolyTerm* next;
};

struct VarItem {
	int varNum;
	int varPow;
	struct VarItem* next;
};