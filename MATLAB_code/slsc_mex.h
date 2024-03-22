#ifndef SLSC_MEX_H
#define SLSC_MEX_H
#ifdef INT_2_BYTES
typedef char int8;
typedef int int16;
typedef long int32;
typedef long long int64;
typedef unsigned char uint8;
typedef unsigned int uint16;
typedef unsigned long uint32;
typedef unsigned long long uint64;
#else
typedef char int8;
typedef short int16;
typedef int int32;
typedef long int64;
typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint32;
typedef unsigned long uint64;
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <string.h>
#include <typeinfo>
#include "mex.h"
#include "blas.h"
#include "computeSTD.h"
#include <omp.h>

// Prototypes of local functions
mxArray *makelagmat(mwSize numrows, mwSize numcols, int maxlag);
//int round(float a);

// Fix for windows
#ifdef _WIN32
#define ssyrk_ ssyrk
#define cherk_ cherk
#define dsyrk_ dsyrk
#define zherk_ zherk
#endif

// Include these after, since they need prototype declaration
#include "rfdata.h"
#include "iqdata.h"



#endif
