struct Polynomial *parsePoly(char *, struct PolynomialSystem *);
struct PolyTerm *parseTerm(char *, int);
struct VarItem *parseVar(char *);
struct PolynomialSystem *buildPolySystem(FILE *, int);
