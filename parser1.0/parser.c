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

struct PolyTerm *parseTerm(char *str, int *variables, int dimension, int line_number) {
  struct PolyTerm *term = (struct PolyTerm *) malloc (sizeof(struct PolyTerm));
  int startSearchIndex=0;
  char *buffer;

  if (str[0] == '[') {
    int indexOfCoeffStart = indexOfStart(str, '[', startSearchIndex);
    int indexOfCoeffEnd = indexOfStart(str, ']', startSearchIndex);
    startSearchIndex = indexOfCoeffEnd;

    int coeffLen = indexOfCoeffEnd - indexOfCoeffStart - 1;

    buffer = (char*) malloc(coeffLen * sizeof(char));
    substring(str, buffer, (indexOfCoeffStart + 1), coeffLen);

    term->coeff = strtof(buffer, NULL);
    free(buffer);
  } else {
    term->coeff = 1.0;
  }

  int size = 0, startIndex = indexOfStart(str,'x', startSearchIndex);

  // get number of variables in monomial
  while (startIndex != -1) {
    size++;
    startIndex = indexOfStart(str, 'x', startIndex+1);
  }

  term->exponents = (int *) malloc (sizeof(int)*dimension);
  for (int i=0; i<dimension; i++)
    term->exponents[i] = 0;
  term->degree = 0;

  //construct the vars array
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

    int dim_var = 0;
    while (item->varNum != variables[dim_var] && dim_var < dimension)
      dim_var++;


    if (dim_var < dimension) {
      if (term->exponents[dim_var] > 0)
        printf("WARNING. Duplicate variables in monomial\n\tVariable: x%d\n\tLine: %d\n", item->varNum, line_number);
      term->exponents[dim_var] += item->varPow;
      term->degree += item->varPow;
    } else {
      printf("ERROR: variable out of dimension bounds\n\tVariable: x%d\n\tLine: %d\n", item->varNum, line_number);
    }

    startIndex = index;
  }

  return term;
}

struct Polynomial *parsePoly(char *str, struct PolynomialSystem *system)
{
  int line_number = system->size+2;
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

    struct PolyTerm *term = parseTerm(trimwhitespace(buffer), system->variables, system->dimension, line_number);
    free(buffer);

    //insert term into polynomial
    if (poly->size==0)
    {
      poly->head = term;
      poly->tail = term;
    } else {
      struct PolyTerm *cmp = poly->head;
      int diff=0;
      for (int i=0; i<poly->size; i++)
      {
        //compare the two monomials according to the monomial ordering
        diff = mono_cmp(term->exponents, cmp->exponents, system->dimension, system->order);

        //debug printing
        //printMonomial2(term, system->variables, system->dimension);
        //printf(" - ");
        //printMonomial2(cmp, system->variables, system->dimension);
        //printf(" = %d\n", diff);

        if ( diff > 0 )
        { //monomial comes before cmp
          if (cmp == poly->head)
            poly->head = term;
          else
          {
            cmp->prev->next = term;
            term->prev = cmp->prev;
          }

          term->next = cmp;
          cmp->prev = term;
          break;
        } else if (diff < 0 && i == poly->size-1 )
        { //monomial comes after the tail
          term->prev = cmp;
          cmp->next = term;
          poly->tail = term;
          break;
        } else if (diff == 0)
        { // monomial is the same as cmp
          printf("\nThere are duplicate monomials in a polynomial.");
          printMonomial2(term, system->variables, system->dimension);
          printf("\nResolving... adding coefficients: %f + %f", term->coeff, cmp->coeff);

          cmp->coeff += term->coeff;
          poly->size--;

          break;
        } else
        { // monomials location has not been found
          cmp = cmp->next;
        }
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

  if (mono_order == 0)
    system->order = grevlex;
  else if (mono_order == 1)
    system->order = grlex;
  else
    system->order = lex;

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

    } else {
      struct Polynomial *poly = parsePoly(str, system);

      if (system->size == 0) {
        system->head = poly;
        system->tail = poly;
      } else {
        poly->prev = system->tail;
        system->tail->next = poly;
        system->tail = poly;
      }

      if (poly->head->degree > system->degree)
        system->degree = poly->head->degree;

    }

    system->size++;
  }

  return system;
}
