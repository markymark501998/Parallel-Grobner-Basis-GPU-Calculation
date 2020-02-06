// builds a macaulay matrix of every monomial up the degree of the polynomial
// limits: left: x1^d, right: xn^d

struct Macaulay {
  enum MonomialOrdering order;
  struct Mat_Monomial *mono_head, *mono_tail;
  int dimension, degree, mono_count, size, *variables;
  struct Monomial **monomials;
  int *monomial_index; // monomial_index[degree] = index in monomials of the first monomial of that degree
  float **m;
};

struct Mat_Monomial {
  struct Monomial *monomial;
  struct Mat_Monomial *next, *prev;
};

struct Macaulay *PolySystem2Macaulay(struct PolynomialSystem *);
struct PolynomialSystem *Macaulay2PolySystem(struct Macaulay *);
void printMacaulay(struct Macaulay *);
int grevlex_mat(struct Mat_Monomial *, struct Mat_Monomial *, int);
int grlex_mat(struct Mat_Monomial *, struct Mat_Monomial *, int);
int lex_mat(struct Mat_Monomial *, struct Mat_Monomial *, int);
