#include <stdio.h>
#include <stdlib.h>
#include "poly.h"
#include "macaulay.h"

int grevlex_mat(struct Mat_Monomial *a, struct Mat_Monomial *b, int d) {
  if (a->monomial->degree > b->monomial->degree)
    return 1;
  else if (b->monomial->degree > a->monomial->degree)
    return -1;

  for (int i=d-1; i>=0; i--) {
    if (a->monomial->exponents[i] < b->monomial->exponents[i])
      return 1;
    else if (b->monomial->exponents[i] < a->monomial->exponents[i])
      return -1;
  }

  return 0;
}

int grlex_mat(struct Mat_Monomial *a, struct Mat_Monomial *b, int d) {
  if (a->monomial->degree > b->monomial->degree)
    return 1;
  else if (b->monomial->degree > a->monomial->degree)
    return -1;

  for (int i=0; i<d; i++) {
    if (a->monomial->exponents[i] > b->monomial->exponents[i])
      return 1;
    else if (b->monomial->exponents[i] > a->monomial->exponents[i])
      return -1;
  }

  return 0;
}

int lex_mat(struct Mat_Monomial *a, struct Mat_Monomial *b, int d) {
  for (int i=0; i<d; i++) {
    if (a->monomial->exponents[i] > b->monomial->exponents[i])
      return 1;
    else if (b->monomial->exponents[i] > a->monomial->exponents[i])
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


struct Macaulay *PolySystem2Macaulay(struct PolynomialSystem *system) {
  struct Macaulay *matrix = (struct Macaulay *) malloc (sizeof(struct Macaulay));

//  set properties
  matrix->order = system->order;
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

      mat_mono->monomial = mono_copy(term->monomial, system->dimension);

/*

      mat_mono->monomial = (struct Monomial *) malloc (sizeof(struct Monomial));

      mat_mono->monomial->degree = term->monomial->degree;

      mat_mono->monomial->exponents = (int *) malloc (sizeof(int)*system->dimension);
      memcpy(mat_mono->monomial->exponents, term->monomial->exponents, sizeof(int)*matrix->dimension);

*/
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
          if (matrix->order == grevlex)
            diff = grevlex_mat(mat_mono, cmp, matrix->dimension);
          else if (matrix->order == grlex)
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

    /*
    matrix->monomials[i] = (struct Monomial *) malloc (sizeof(struct Monomial));

    matrix->monomials[i]->degree = mono->monomial->degree;
    matrix->monomials[i]->exponents = (int *) malloc (sizeof(int)*matrix->dimension);
    memcpy(matrix->monomials[i]->exponents, mono->monomial->exponents, sizeof(int)*matrix->dimension);
    */

    matrix->monomials[i] = mono->monomial;

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
      int search_index = matrix->monomial_index[matrix->degree-term->monomial->degree];

      while((mono_eq(matrix->monomials[search_index]->exponents, term->monomial->exponents, matrix->dimension) == 0) && matrix->monomials[search_index]->degree == term->monomial->degree) {
        matrix->m[i][search_index] = 0;
        search_index++;
      }

      if (matrix->monomials[search_index]->degree == term->monomial->degree) {
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

struct PolynomialSystem *Macaulay2PolySystem(struct Macaulay *matrix) {
  // initialize system
  struct PolynomialSystem *system = (struct PolynomialSystem *) malloc (sizeof(struct PolynomialSystem));

  // set properties
  system->order = matrix->order;
  system->dimension = matrix->dimension;
  system->degree = matrix->degree;
  system->size = matrix->size;
  system->variables = (int *) malloc (sizeof(int)*system->dimension);
  for(int i=0; i<system->dimension; i++)
    system->variables[i] = matrix->variables[i];

  // build polynomials
  for (int r = 0; r < system->size; r++) {
    // initialize poly
    struct Polynomial *poly = (struct Polynomial *) malloc (sizeof(struct Polynomial));

    // set properties
    poly->size = 0;

    int c = 0;
    while (c < matrix->mono_count) {
      if (matrix->m[r][c] != 0) {
        // create term
        struct PolyTerm *term = (struct PolyTerm *) malloc (sizeof(struct PolyTerm));

        term->coeff = matrix->m[r][c];

        term->monomial = mono_copy(matrix->monomials[c], matrix->dimension);

        if (poly->size == 0) {
          poly->head = term;
          poly->tail = term;
        } else {
          poly->tail->next = term;
          term->prev = poly->tail;
          poly->tail = term;
        }

        poly->size++;
      }

      c++;
    }

    if (r == 0) {
      system->head = poly;
      system->tail = poly;
    } else {
      system->tail->next = poly;
      poly->prev = system->tail;
      system->tail = poly;
    }
  }

  return system;
}



void printMacaulay(struct Macaulay *matrix) {
  struct Mat_Monomial *mono = matrix->mono_head;
  for (int i=0; i<matrix->mono_count; i++) {
    printf("  ");
    for(int j=0; j<matrix->dimension; j++) {
      if(mono->monomial->exponents[j] > 0)
        printf("x%d^%d", matrix->variables[j], mono->monomial->exponents[j]);
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
