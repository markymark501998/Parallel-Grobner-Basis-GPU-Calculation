#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "poly.h"
#include "macaulay.h"

int grevlex_mat(struct Mat_Monomial *a, struct Mat_Monomial *b, int d) {
  if (a->total > b->total)
    return 1;
  else if (b->total > a->total)
    return -1;

  for (int i=d-1; i>=0; i--) {
    if (a->exponents[i] < b->exponents[i])
      return 1;
    else if (b->exponents[i] < a->exponents[i])
      return -1;
  }

  return 0;
}

int grlex_mat(struct Mat_Monomial *a, struct Mat_Monomial *b, int d) {
  if (a->total > b->total)
    return 1;
  else if (b->total > a->total)
    return -1;

  for (int i=0; i<d; i++) {
    if (a->exponents[i] > b->exponents[i])
      return 1;
    else if (b->exponents[i] > a->exponents[i])
      return -1;
  }

  return 0;
}

int lex_mat(struct Mat_Monomial *a, struct Mat_Monomial *b, int d) {
  for (int i=0; i<d; i++) {
    if (a->exponents[i] > b->exponents[i])
      return 1;
    else if (b->exponents[i] > a->exponents[i])
      return -1;
  }

  return 0;
}

int mono_eq(int *exp_a, int *exp_b, int d) {
  int match = 1;
  for (int i=0; i<d; i++)
  {
    if (exp_a[i] != exp_b[i])
    {
      match = 0;
      break;
    }
  }

  return match;
}


struct Macaulay *buildMacaulay(struct PolynomialSystem *system, int mono_order) {
  struct Macaulay *matrix = (struct Macaulay *) malloc (sizeof(struct Macaulay));

//  set properties
  matrix->dimension = system->dimension;
  matrix->degree = system->degree;
  matrix->mono_count = 0;
  matrix->size = system->size;
  matrix->variables = (int *) malloc (sizeof(int)*matrix->dimension);
  for(int i=0; i<matrix->dimension; i++)
    matrix->variables[i] = system->variables[i];

// build list of monomials
  struct Polynomial * poly = system->head;

  for (int o=0; o<system->size; o++) {
    struct PolyTerm *term = poly->head;

    for (int p=0; p<poly->size; p++) {

      //build Mat_Monomial
      struct Mat_Monomial *mat_mono = (struct Mat_Monomial *) malloc (sizeof(struct Mat_Monomial));
      mat_mono->total = term->degree;
      mat_mono->exponents = (int *) malloc (sizeof(int)*system->dimension);
      for (int i=0; i<system->dimension; i++)
        mat_mono->exponents[i] = term->exponents[i];

      // add monomial to the monomial list
      if (matrix->mono_count == 0) {
        // first monomial, initialize the matrix headers
        matrix->mono_head = mat_mono;
        matrix->mono_tail = mat_mono;
        matrix->mono_count++;
      } else {
        // check if this is the location of the monomial
        struct Mat_Monomial *cmp = matrix->mono_head;
        for(int j=0; j<matrix->mono_count; j++) {
          int diff=0;
          if (mono_order == 0)
            diff = grevlex_mat(mat_mono, cmp, matrix->dimension);
          else if (mono_order == 1)
            diff = grlex_mat(mat_mono, cmp, matrix->dimension);
          else
            diff = lex_mat(mat_mono, cmp, matrix->dimension);

          if (diff == 0) {
            break;
          } else if (diff > 0) {
            if (cmp == matrix->mono_head) {
              matrix->mono_head = mat_mono;
            } else {
              cmp->prev->next = mat_mono;
              mat_mono->prev = cmp->prev;
            }
            cmp->prev = mat_mono;
            mat_mono->next = cmp;
            matrix->mono_count++;
            break;
          } else if (cmp == matrix->mono_tail && diff < 0) {
            cmp->next = mat_mono;
            mat_mono->prev = cmp;
            matrix->mono_tail = mat_mono;
            matrix->mono_count++;
            break;
          }

          cmp = cmp->next;
        }
      }
      // the next term will already be ordered < mat_mono
      // previousMonomial = mat_mono;
      term = term->next;
    }
    poly = poly->next;
  }

  // create the monomial_index
  matrix->monomial_index = (int *) malloc (sizeof(int)*matrix->degree);
  for (int i = 0; i<matrix->degree; i++)
    matrix->monomial_index[i] = -1;

  // build the monomial array and find degree start indices
  matrix->monomials = (struct Monomial **) malloc (sizeof(struct Monomial *)*matrix->mono_count);

  struct Mat_Monomial *mono = matrix->mono_head;

  for (int i = 0; i<matrix->mono_count; i++) {
    matrix->monomials[i] = (struct Monomial *) malloc (sizeof(struct Monomial));

    matrix->monomials[i]->degree = mono->total;
    matrix->monomials[i]->exponents = (int *) malloc (sizeof(int)*matrix->dimension);
    memcpy(matrix->monomials[i]->exponents, mono->exponents, sizeof(int)*matrix->dimension);

    if (i == 0)
      matrix->monomial_index[i] = i;
    else {
      if (matrix->monomials[i]->degree < matrix->monomials[i-1]->degree)
        matrix->monomial_index[matrix->degree-matrix->monomials[i]->degree] = i;
    }

    mono = mono->next;
  }

  //  printing each monomial_index for validation
/*
  for(int i = 0; i<matrix->degree; i++) {
    if(matrix->monomial_index[i] == -1)
      printf("\tNULL");
    else {
      printf("\t");
      printMonomial(matrix->monomials[matrix->monomial_index[i]]->exponents, matrix->variables, matrix->dimension);
    }
  }
  printf("\n");
*/

  // build float coeff matrix -- rows = polynomial, cols = monomial
  matrix->m = (float **) malloc (sizeof(float *)*system->size);

  poly = system->head;
  for (int i=0; i<system->size; i++)
  {

    // build row of coeff matrix
    matrix->m[i] = (float *) malloc (sizeof(float *)*matrix->mono_count);

    struct Mat_Monomial *mat_mono;
    struct PolyTerm *term = poly->head;
    int column = 0;
    for (int j=0; j<poly->size; j++)
    {
      int search_index = matrix->monomial_index[matrix->degree-term->degree];

      while((mono_eq(matrix->monomials[search_index]->exponents, term->exponents, matrix->dimension) == 0) && matrix->monomials[search_index]->degree == term->degree) {
        matrix->m[i][search_index] = 0;
        search_index++;
      }

      if (matrix->monomials[search_index]->degree == term->degree) {
        matrix->m[i][search_index] = term->coeff;
      } else {
        printf("OHNO\n");
      }

      term = term->next;
    }
    poly = poly->next;
  }

  return matrix;
}



void printMacaulay(struct Macaulay *matrix) {
  struct Mat_Monomial *mono = matrix->mono_head;
  for (int i=0; i<matrix->mono_count; i++) {
    printf("  ");
    for(int j=0; j<matrix->dimension; j++) {
      if(mono->exponents[j] > 0)
        printf("x%d^%d", matrix->variables[j], mono->exponents[j]);
    }
    mono = mono->next;
  }
  printf("\n");

  for (int i=0; i<matrix->size; i++) {
    for (int j=0; j<matrix->mono_count; j++)
      printf("  %6.4f", matrix->m[i][j]);
    printf("\n");
  }

}
