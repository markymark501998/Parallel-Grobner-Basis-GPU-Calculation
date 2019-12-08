#include <stdio.h>
#include <stdlib.h>
#include "poly.h"

void printPolySystem(struct PolynomialSystem *system) {
  int i;
  printf("Dimensions: %d\n", system->dimension);
  printf("Degree: %d\n", system->degree);
  printf("x%d", system->variables[0]);
  for ( i=1; i<system->dimension; i++ ) {
    printf(", x%d", system->variables[i]);
  }
  printf("\n");

  struct Polynomial *poly = system->head;
  for(i=0; i<system->size; i++) {
    printf("%d : ", i);
    printPoly(poly);

    poly = poly->next;
  }
}

void printTerm(struct PolyTerm *term) {
  printf("%f", term->coeff);

  for (int i = 0; i < term->monomial->num_vars; i++) {
    printf("*x%d^%d", term->monomial->vars[i]->varNum, term->monomial->vars[i]->varPow);
  }
}

void printMonomial(struct Monomial *mono) {
  for (int i = 0; i < mono->num_vars; i++) {
    if (i != 0)
      printf("*");
    printf("x%d^%d", mono->vars[i]->varNum, mono->vars[i]->varPow);
  }
}

void printPoly(struct Polynomial *poly) {
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

int totalDegree(struct Monomial *term) {
  int total = 0;

  for (int i = 0; i < term->num_vars; i++)
    total += term->vars[i]->varPow;

  return total;
}

// Outputs: 1 if a > b, -1 if a < b, 0 if equal according to grevlex,
//  max(totalDegree), if equal, least-powered right-most variable
int grevlex_cmp(struct Monomial *a, struct Monomial *b) {
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
int grlex_cmp(struct Monomial *a, struct Monomial *b) {
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
int lex_cmp(struct Monomial *a, struct Monomial *b) {
  if (a->vars[0]->varNum < b->vars[0]->varNum)
    return 1;
  else if (a->vars[0]->varNum > b->vars[0]->varNum)
    return -1;
  else
    return 0;
}
