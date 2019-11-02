#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utility.h"

void myFunc(void)
{
    printf("Body of myFunc function.\n");
}

int indexOf(char *str, char c)
{
	int i;
	for (i = 0; i < strlen(str); i++)
	{
		if (str[i] == c)
			return i;
	}

	return -1;
}

int indexOfStart(char *str, char c, int start)
{
	int i;
	for (i = start; i < strlen(str); i++)
	{
		if (str[i] == c)
			return i;
	}

	return -1;
}

void substring(char s[], char sub[], int p, int l) {
   int c = 0;
   
   while (c < l) {
      sub[c] = s[p+c];
      c++;
   }
   sub[c] = '\0';
}