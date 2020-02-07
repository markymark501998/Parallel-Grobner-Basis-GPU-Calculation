#include <stdio.h>
#include <stdlib.h>
#include "poly.h"
#include "macaulay.h"
#include "f45.h"

int poly_cmp( const void * a, const void * b ) {
  struct Polynomial *x = (struct Polynomial *)a, *y = (struct Polynomial *)b;
  return x->head->monomial->degree - y->head->monomial->degree;
}

struct PolynomialSystem *f45(struct PolynomialSystem *input) {
  // ===========================================
  // sort input system by total degree
  // ===========================================

  //convert linked list to array
  struct Polynomial **ps = (struct Polynomial **) malloc (sizeof(struct Polynomial *)*input->size);
  struct Polynomial *p = input->head;
  for (int i = 0; i < input->size; i++) {
    ps[i] = p;
    p = p->next;
  }

  qsort(ps, input->size, sizeof(struct Polynomial *), poly_cmp);

  // convert sorted array back to linked list
  input->head = ps[0];
  input->tail = ps[input->size-1];
  struct Polynomial *poly = input->head;
  for (int i = 1; i < input->size; i++) {
    poly->next = ps[i];
    ps[i]->prev = poly;
  }

  // ===========================================
  // declare global variables
  // ===========================================
  int *G;
  struct LabelledPolynomial **L;


  return (struct PolynomialSystem *)0;
}
