#include <stdio.h>
#include <stdlib.h>
#include "poly.h"

void printPolySystem2(struct PolynomialSystem *system) {
  int i;
  printf("Dimensions: %d\n", system->dimension);
  printf("Degree: %d\n", system->degree);
  printf("Order: ");
  if (system->order == grevlex)
    printf("grevlex\n");
  else if (system->order == grlex)
    printf("grlex\n");
  else if (system->order == lex)
    printf("lex\n");

  printf("x%d", system->variables[0]);
  for ( i=1; i<system->dimension; i++ ) {
    printf(", x%d", system->variables[i]);
  }
  printf("\n");

  struct Polynomial *poly = system->head;
  for(i=0; i<system->size; i++) {
    printf("%d : ", i);
    struct PolyTerm *term = poly->head;
    for (int j=0; j<poly->size; j++) {
      if (j > 0)
        printf(" + ");
      printf("%f*", term->coeff);
      printMonomial2(term, system->variables, system->dimension);
      term = term->next;
    }
    printf("\n");
    poly = poly->next;
  }
}

void printMonomial2(struct PolyTerm *term, int *variables, int dimension) {
  int count = 0;
  for (int i=0; i<dimension; i++) {
    if (term->monomial->exponents[i] > 0) {
      if (count>0)
        printf("*");
      printf("x%d^%d", variables[i], term->monomial->exponents[i]);
      count++;
    }
  }
}

void printMonomial(int *deg, int *var, int n) {
  for(int i = 0; i<n; i++) {
    if (deg[i]>0)
      printf("x%d^%d", var[i], deg[i]);
  }
}

/*
 * General monomial compare
 * order = { 0: grevlex, 1: grlex, 2: lex }
 * returns:
 *  positive: a > b
 *  negative: a < b
 *  0: a = b
 */
int mono_cmp(int *exp_a, int *exp_b, int d, enum MonomialOrdering order) {
  int deg_a = 0, deg_b = 0;
  if (order == grevlex || order == grlex){
    for (int i=0; i<d; i++) {
      deg_a += exp_a[i];
      deg_b += exp_b[i];
    }
  }

  //printf("\t%d - %d =%d\n",deg_a, deg_b, deg_a-deg_b);
  
  if ((order == grevlex || order == grlex) && deg_a-deg_b != 0)
    return deg_a-deg_b;
  else if (order == grlex || order == lex) {
    for (int i=0; i<d; i++)
      if (exp_a[i]-exp_b[i] != 0)
        return exp_a[i]-exp_b[i];
  } else if (order == grevlex)
    for (int i=d-1; i>=0; i--) {
      if (exp_b[i]-exp_a[i] != 0)
        return exp_b[i]-exp_a[i];
  }
  return 0;
}
