#include <stdio.h>
#include <stdlib.h>
#include "poly.h"

void printTerm(struct PolyTerm *term) {
  printf("%f", term->coeff);

  for (int i = 0; i < term->num_vars; i++) {
    printf("*x%d^%d", term->vars[i]->varNum, term->vars[i]->varPow);
  }
}

void printPoly(struct Polynomial *poly) {
  //Print the Polynomial
  struct PolyTerm* term;
  term = poly->head;

  while (term != poly->tail) {
    printTerm(term);
    printf(" + ");
    term = term->next;
  }

  printTerm(poly->tail);
  printf("\n");
}

int totalDegree(struct PolyTerm *term) {
  int total = 0;

  for (int i = 0; i < term->num_vars; i++)
    total += term->vars[i]->varPow;

  return total;
}

// Outputs: 1 if a > b, -1 if a < b, 0 if equal according to grevlex,
//  max(totalDegree), if equal, least-powered right-most variable
int grevlex_cmp(struct PolyTerm *a, struct PolyTerm *b) {
  int tda = totalDegree(a);
  int tdb = totalDegree(b);

  if (tda > tdb)
    return 1;
  else if (tda < tdb)
    return -1;
  else if (a->vars[a->num_vars-1]->varNum < b->vars[b->num_vars-1]->varNum)
    return 1;
  else if (a->vars[a->num_vars-1]->varNum > b->vars[b->num_vars-1]->varNum)
    return -1;
  else
    return 0;
}

// Outputs: 1 if a > b, -1 if a < b, 0 if equal according to grevlex,
//  max(totalDegree), if equal, least-powered right-most variable
int grlex_cmp(struct PolyTerm *a, struct PolyTerm *b) {
  int tda = totalDegree(a);
  int tdb = totalDegree(b);

  if (tda > tdb)
    return 1;
  else if (tda < tdb)
    return -1;
  else if (a->vars[0]->varNum < b->vars[0]->varNum)
    return 1;
  else if (a->vars[0]->varNum > b->vars[0]->varNum)
    return -1;
  else
    return 0;
}

// Outputs: 1 if a > b, -1 if a < b, 0 if equal according to grevlex,
//  max(totalDegree), if equal, least-powered right-most variable
int lex_cmp(struct PolyTerm *a, struct PolyTerm *b) {
  if (a->vars[0]->varNum < b->vars[0]->varNum)
    return 1;
  else if (a->vars[0]->varNum > b->vars[0]->varNum)
    return -1;
  else
    return 0;
}
