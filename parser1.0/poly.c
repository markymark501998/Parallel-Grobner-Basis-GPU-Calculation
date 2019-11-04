#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utility.h"
#include "poly.h"

struct Polynomial* parsePoly(char *str)
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
    currentTerm = (struct PolyTerm*) malloc (sizeof(struct PolyTerm));

    if(firstTermFlag == 1)
    {
      firstTermFlag = 0;
      poly->head = currentTerm;
    }
    else
    {
      if(str[startSearchIndex] != '+' && str[startSearchIndex+1] != '+')
      {
        if(str[startSearchIndex] == '\n' || str[startSearchIndex+1] == '\n')
        {
          break;
        }
      }
      currentTerm->prev = poly->tail;
      poly->tail->next = currentTerm;
    }

    poly->tail = currentTerm;

    int indexOfCoeffStart = indexOfStart(str, '[', startSearchIndex);
    int indexOfCoeffEnd = indexOfStart(str, ']', startSearchIndex);
    startSearchIndex = indexOfCoeffEnd;

    int coeffLen = indexOfCoeffEnd - indexOfCoeffStart - 1;

    buffer = (char*) malloc(coeffLen * sizeof(char));
    substring(str, buffer, (indexOfCoeffStart + 1), coeffLen);

    currentTerm->coeff = strtof(buffer, NULL);

    int doneFlag = 0;
    int firstFlag = 0;

    struct VarItem* nextItem;

    while (doneFlag == 0) {
      startSearchIndex++;
      if(str[startSearchIndex] != 'x') {
        if(str[startSearchIndex] != '*' && str[startSearchIndex + 1] != 'x') {
          break;
        }
      }

      nextItem = (struct VarItem*) malloc (sizeof(struct VarItem));

      if(firstFlag == 0) {
        firstFlag = 1;

        currentTerm->head = nextItem;
      }
      else {
        nextItem->prev = currentTerm->tail;
        currentTerm->tail->next = nextItem;
      }

      currentTerm->tail = nextItem;

      int bracketStart = indexOfStart(str, '[', startSearchIndex);
      int bracketEnd = indexOfStart(str, ']', startSearchIndex);
      startSearchIndex = bracketEnd + 1;
      int numLength = bracketEnd - bracketStart - 1;

      free(buffer);
      buffer = (char*) malloc(numLength * sizeof(char));
      substring(str, buffer, (bracketStart + 1), numLength);
      nextItem->varNum = atoi(buffer);

      if(str[startSearchIndex] != '^') {
        break;
      }

      bracketStart = indexOfStart(str, '[', startSearchIndex);
      bracketEnd = indexOfStart(str, ']', startSearchIndex);
      startSearchIndex = bracketEnd + 1;
      numLength = bracketEnd - bracketStart - 1;

      free(buffer);
      buffer = (char*) malloc(numLength * sizeof(char));
      substring(str, buffer, (bracketStart + 1), numLength);
      nextItem->varPow = atoi(buffer);

      if((str[startSearchIndex-1] == ']' && str[startSearchIndex] == '\0')
      || (str[startSearchIndex-1] == ']' && (str[startSearchIndex] == '+' || str[startSearchIndex+1] == '+'))) {
        doneFlag = 1;
      }
    }

    if(str[startSearchIndex+1] != '[') {
      break;
    }
  }

  //At this point the variable "poly" has a linked list of terms (not ordered)
  //for a single polynomial (aka: a single row of the input text file)

  return poly;
}

void printTerm(struct PolyTerm *term) {
  printf("%f*", term->coeff);

  struct VarItem* variable;
  variable = term->head;

  printf("x[%d]^[%d]", variable->varNum, variable->varPow);

  while ((variable=variable->next) != NULL) {
    printf("*x[%d]^[%d]", variable->varNum, variable->varPow);
  }
}

void printPoly(struct Polynomial *poly) {
  //Print the Polynomial
  struct PolyTerm* term;
  term = poly->head;

  while (term != NULL) {
    printTerm(term);

    if((term = term->next) != NULL) {
      printf(" + ");
    } else {
      printf("\n");
    }
  }
}

int totalDegree(struct PolyTerm *term) {
  struct VarItem *var = term->head;
  int total = var->varPow;

  while((var=var->next) != NULL)
    total += var->varPow;

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
  else {
    struct VarItem *right_a = a->head, *right_b = b->head, *cmp;

    // get the right-most term from a
    while((cmp=right_a->next) != NULL)
      if (cmp->varNum > right_a->varNum)
        right_a = cmp;

    // get the right-most term from b
    while((cmp=right_b->next) != NULL)
      if (cmp->varNum > right_b->varNum)
        right_b = cmp;

    if (right_a->varNum > right_b->varNum)
      return 1;
    else if (right_b->varNum > right_a->varNum)
      return -1;
    else {
      if (right_a->varPow < right_b->varPow)
        return 1;
      else if (right_b->varPow < right_a->varPow)
        return -1;
      else
        return 0;
    }
  }
}

void sortTerm(struct PolyTerm *term) {
  struct PolyTerm newTerm;

  struct VarItem *lv = term->head, *tv = term->tail , *cmp = lv->next;
  //find the leading variable
  do {
    if (lv->varNum > cmp->varNum)
      lv = cmp;
  } while((cmp=cmp->next) != NULL);
}

struct PolyTerm *findLargestTerm(struct Polynomial *poly, int startIndex) {
  struct PolyTerm *max = poly->head, *cmp = poly->head;

  for(int i=0; i<startIndex; i++) {
    max = max->next;
    cmp = cmp->next;
  }

  while(cmp != poly->tail){
    printTerm(max);
    printf(" vs ");
    printTerm(cmp);
    printf("\n");
    if (grevlex_cmp(max, cmp) < 0) {
      max = cmp;
    }
    printf("Next\n");

    if (cmp == poly->tail || cmp->next == cmp)
      break;

    cmp = cmp->next;
  }

  return max;
}

void swap(struct PolyTerm *a, struct PolyTerm *b) {
  // move b into a's adjacent nodes
  if (a->prev != NULL)
    a->prev->next = b;
  if (a->next != NULL)
    a->next->prev = b;

  // move a into b's adjacent nodes
  if (b->prev != NULL)
    b->prev->next = a;
  if (b->next != NULL)
    b->next->prev = a;

  // swap a and b's next
  struct PolyTerm *tmp = a->next;
  a->next = b->next;
  b->next = tmp;

  // swap a and b's prev
  tmp = a->prev;
  a->prev = b->prev;
  b->prev = tmp;
}

void sortPoly(struct Polynomial *poly) {
  int sorted = 0;
  struct PolyTerm *largest, *sorting = poly->head;

  //selection sort
  while (sorting != poly->tail) {
    printTerm(sorting);
    printf(" will be swapping with...\n\t");
    largest = findLargestTerm(poly, sorted);
    printTerm(largest);
    printf("\n");

    if(sorting != largest)
      swap(largest, sorting);

    if (sorting == poly->head)
      poly->head = largest;
    if (largest == poly->tail)
      poly->tail = sorting;

    sorted++;
    sorting = sorting->next;
  }

  poly->tail = sorting;
}
