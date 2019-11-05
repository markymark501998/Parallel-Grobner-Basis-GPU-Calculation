#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "poly.h"
#include "utility.h"

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


  int size = 0, startIndex = indexOfStart(str,'x', startSearchIndex);

  // calculate size of array: num_vars
  while (startIndex != -1) {
    size++;
    startIndex = indexOfStart(str, 'x', startIndex+1);
  }

  term->num_vars=size;

  //construct the vars array
  term->vars = (struct VarItem **) malloc (size * sizeof(struct VarItem *));
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
        term->vars[j] = item;
        break;
      } else if (item->varNum == term->vars[j-1]->varNum) {
        printf("\n\nWARNING: Input contains duplicate variables in a monomial. x%d^%d and x%d^%d\n\n", term->vars[j-1]->varNum, term->vars[j-1]->varPow, item->varNum, item->varPow);
        term->vars[j] = item;
        break;
      } else if (item->varNum < term->vars[j-1]->varNum) {
        term->vars[j] = term->vars[j-1];
      } else {
        term->vars[j] = item;
        break;
      }
    }

    startIndex = index;
  }

  return term;
}

struct Polynomial *parsePoly(char *str, int mono_order)
{
  char* buffer;
  int startSearchIndex = 0;

  struct Polynomial *poly;
  poly = (struct Polynomial*) malloc (sizeof(struct Polynomial));
  int doneParsingPoly = 0;
  int firstTermFlag = 1;

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
    if (firstTermFlag==1) {
      poly->head = term;
      poly->tail = term;
      firstTermFlag=0;
    } else {
      struct PolyTerm *cmp = poly->head;

      while (cmp != poly->tail) {
        if ((mono_order==0 && grevlex_cmp(term, cmp) > 0) ||
          (mono_order==1 && grlex_cmp(term, cmp) > 0) ||
          (mono_order==2 && lex_cmp(term, cmp) > 0) )
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

      if ((mono_order==0 && grevlex_cmp(term, cmp) > 0) ||
        (mono_order==1 && grlex_cmp(term, cmp) > 0) ||
        (mono_order==2 && lex_cmp(term, cmp) > 0) )
      {
        if (cmp == poly->head)
          poly->head = term;
        else
          cmp->prev->next = term;

        term->next = cmp;
        cmp->prev = term;
      } else if (cmp == poly->tail && (
        (mono_order==0 && grevlex_cmp(term, cmp) < 0) ||
        (mono_order==1 && grlex_cmp(term, cmp) < 0) ||
        (mono_order==2 && lex_cmp(term, cmp) < 0) ) )
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

    startSearchIndex = termBreakIndex+1;
  }

  //At this point the variable "poly" has a linked list of terms (not ordered)
  //for a single polynomial (aka: a single row of the input text file)

  return poly;
}
