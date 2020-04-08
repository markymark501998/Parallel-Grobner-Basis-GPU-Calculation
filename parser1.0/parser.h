struct Polynomial *parsePoly(char *, int);
struct PolyTerm *parseTerm(char *);
struct VarItem *parseVar(char *);
struct PolynomialSystem *buildPolySystem(FILE *, int);
