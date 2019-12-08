#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "poly.h"
#include "utility.h"
#define MAXCHAR 100000

struct VarItem *parseVar(char *str) {
  struct VarItem* var;
  char *buffer;
  int startSearchIndex = 0;

  var = (struct VarItem*) malloc (sizeof(struct VarItem));

  int bracketStart = indexOfStart(str, '[', startSearchIndex);
  int bracketEnd = indexOfStart(str, ']', startSearchIndex);
  startSearchIndex = bracketEnd + 1;
  int numLength = bracketEnd - bracketStart - 1;

  buffer = (char*) malloc(numLength * sizeof(char));
  substring(str, buffer, (bracketStart + 1), numLength);
  var->varNum = atoi(buffer);
  free(buffer);

  bracketStart = indexOfStart(str, '[', startSearchIndex);
  bracketEnd = indexOfStart(str, ']', startSearchIndex);
  startSearchIndex = bracketEnd + 1;
  numLength = bracketEnd - bracketStart - 1;

  buffer = (char*) malloc(numLength * sizeof(char));
  substring(str, buffer, (bracketStart + 1), numLength);
  var->varPow = atoi(buffer);
  free(buffer);

  return var;
}

int var_compare( const void* a, const void* b)
{
     struct VarItem *var_a = (struct VarItem *)a;
     struct VarItem *var_b = (struct VarItem *)b;

     if ( var_a->varNum == var_b->varNum ) return 0;
     else if ( var_a->varNum < var_b->varNum ) return -1;
     else return 1;
}

struct PolyTerm *parseTerm(char *str) {
  struct PolyTerm *term = (struct PolyTerm *) malloc (sizeof(struct PolyTerm));
  int startSearchIndex=0;
  char *buffer;
  int indexOfCoeffStart = indexOfStart(str, '[', startSearchIndex);
  int indexOfCoeffEnd = indexOfStart(str, ']', startSearchIndex);
  startSearchIndex = indexOfCoeffEnd;

  int coeffLen = indexOfCoeffEnd - indexOfCoeffStart - 1;

  buffer = (char*) malloc(coeffLen * sizeof(char));
  substring(str, buffer, (indexOfCoeffStart + 1), coeffLen);

  term->coeff = strtof(buffer, NULL);
  free(buffer);

  term->monomial = (struct Monomial *) malloc (sizeof(struct Monomial));
  int size = 0, startIndex = indexOfStart(str,'x', startSearchIndex);

  // calculate size of array: num_vars
  while (startIndex != -1) {
    size++;
    startIndex = indexOfStart(str, 'x', startIndex+1);
  }

  term->monomial->num_vars=size;

  //construct the vars array
  term->monomial->vars = (struct VarItem **) malloc (size * sizeof(struct VarItem *));
  startIndex = indexOfStart(str, 'x', startSearchIndex);

  for (int i = 0; i < size; i++) {
    int length, index = indexOfStart(str, 'x', startIndex+1);
    if (i == size-1)
      length = strlen(str)-startIndex;
    else
      length = index-startIndex-1;
    buffer = (char *) malloc (length*sizeof(char));
    substring(str, buffer, startIndex, length);

    struct VarItem *item = parseVar(buffer);
    free(buffer);
    for (int j=i; j>=0; j--) {
      if (j == 0) {
        term->monomial->vars[j] = item;
        break;
      } else if (item->varNum == term->monomial->vars[j-1]->varNum) {
        printf("\n\nWARNING: Input contains duplicate variables in a monomial. x%d^%d and x%d^%d\n\n", term->monomial->vars[j-1]->varNum, term->monomial->vars[j-1]->varPow, item->varNum, item->varPow);
        term->monomial->vars[j] = item;
        break;
      } else if (item->varNum < term->monomial->vars[j-1]->varNum) {
        term->monomial->vars[j] = term->monomial->vars[j-1];
      } else {
        term->monomial->vars[j] = item;
        break;
      }
    }

    startIndex = index;
  }

  qsort(term->monomial->vars, term->monomial->num_vars, sizeof(struct VarItem *), var_compare);

  return term;
}

struct Polynomial *parsePoly(char *str, int mono_order)
{
  char* buffer;
  int startSearchIndex = 0;

  struct Polynomial *poly;
  poly = (struct Polynomial*) malloc (sizeof(struct Polynomial));
  poly->size = 0;
  int doneParsingPoly = 0;

  struct PolyTerm* currentTerm;

  while(doneParsingPoly == 0)
  {
    int termBreakIndex = indexOfStart(str, '+', startSearchIndex);
    int termLength;
    if (termBreakIndex == -1){
      termLength = strlen(str) - startSearchIndex;
      doneParsingPoly = 1;
    }else
      termLength = termBreakIndex - startSearchIndex;

    buffer = (char*) malloc(termLength * sizeof(char));
    substring(str, buffer, startSearchIndex, termLength);

    struct PolyTerm *term = parseTerm(buffer);
    free(buffer);

    //insert term into polynomial
    if (poly->size==0) {
      poly->head = term;
      poly->tail = term;
    } else {
      struct PolyTerm *cmp = poly->head;
      int diff;
      if (mono_order==0)
        diff = grevlex_cmp(term->monomial, cmp->monomial);
      else if (mono_order==1)
        diff = grlex_cmp(term->monomial, cmp->monomial);
      else
        diff = lex_cmp(term->monomial, cmp->monomial);

      while (cmp != poly->tail) {
        if (diff > 0)
        {
          if(cmp == poly->head)
            poly->head = term;
          else
            cmp->prev->next = term;

          term->next = cmp;
          cmp->prev = term;

          break;
        }
        cmp = cmp->next;
      }

      if ( diff > 0 )
      {
        if (cmp == poly->head)
          poly->head = term;
        else
          cmp->prev->next = term;

        term->next = cmp;
        cmp->prev = term;
      } else if (cmp == poly->tail && diff < 0 )
      {
        term->prev = cmp;
        cmp->next = term;
        poly->tail = term;
      } else {
        printf("\nThere are duplicate monomials in a polynomial.");
        printTerm(term);
        printf(" and ");
        printTerm(cmp);
        printf("\n\n");
        term->prev = cmp;
        cmp->next = term;
        poly->tail = term;
      }
    }
    poly->size++;
    startSearchIndex = termBreakIndex+1;
  }

  //At this point the variable "poly" has a linked list of terms (not ordered)
  //for a single polynomial (aka: a single row of the input text file)

  return poly;
}

int int_compare ( const void *aa, const void *bb ) {
  const int *a = aa, *b = bb;
  return (*a < *b) ? -1 : (*a > *b);
}

struct PolynomialSystem *buildPolySystem(FILE *f, int mono_order) {
  char str[MAXCHAR];
  struct PolynomialSystem *system = (struct PolynomialSystem*) malloc (sizeof(struct PolynomialSystem));

  system->size = -1;

  while (fgets(str, MAXCHAR, f) != NULL) {

    // if first line, get the dimension/variables of the system
    // else build polynomial and add it to the system
    if (system->size == -1) {
      int dim = 0;
      int startSearchIndex=0;
      char *buffer;
      int varLen, indexOfVarEnd, indexOfVarStart = indexOfStart(str, '[', startSearchIndex);

      while (indexOfVarStart != -1 ) {
        dim++;
        startSearchIndex = indexOfVarStart + 1;
        indexOfVarStart = indexOfStart(str, '[', startSearchIndex);
      }

      system->dimension = dim;
      system->variables = (int *) malloc (dim*sizeof(int));
      system->degree = 0;

      startSearchIndex = 0;

      for (int i=0; i<system->dimension; i++) {
        indexOfVarStart = indexOfStart(str, '[', startSearchIndex);
        indexOfVarEnd = indexOfStart(str, ']', startSearchIndex);
        varLen = indexOfVarEnd - indexOfVarStart - 1;

        buffer = (char*) malloc(varLen * sizeof(char));
        substring(str, buffer, (indexOfVarStart + 1), varLen);

        system->variables[i] = atoi(buffer);
        free(buffer);

        startSearchIndex = indexOfVarEnd + 1;
      }

      qsort(system->variables, system->dimension, sizeof(int), int_compare);

    } else {
      struct Polynomial *poly = parsePoly(str, mono_order);

      if (system->size == 0) {
        system->head = poly;
        system->tail = poly;
      } else {
        poly->prev = system->tail;
        system->tail->next = poly;
        system->tail = poly;
      }

      int d = totalDegree(poly->head->monomial);
      if (d > system->degree)
        system->degree = d;

      struct PolyTerm *term = poly->head;
      for (int i=0; i<poly->size; i++) {
        struct Monomial *mono = term->monomial;
        for (int j=0; j<mono->num_vars; j++) {
          int matched = 0;
          for (int k=0; k<system->dimension; k++) {
            if (mono->vars[j]->varNum == system->variables[k]) {
              matched = 1;
              break;
            }
          }
          if (matched == 0) {
            printf("WARNING: Variable number does not exist in defined dimesion\n\tLine: %d\n\tMonomial: ", i+4);
            printTerm(term);
            printf("\n");
          }
        }
        term = term->next;
      }
    }

    system->size++;
  }

  return system;
}
