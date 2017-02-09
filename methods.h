#ifndef METHODS_H
#define METHODS_H
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
//#include <random>
//#include <iostream>
#include <math.h>
//using namespace std;
#define LIBOCAS_PLUS_INF (-log(0.0))
#define LIBOCAS_CALLOC(x,y) calloc(x,y)
#define LIBOCAS_FREE(x) free(x)
//#define LIBOCAS_MIN(A,B) ((A) > (B) ? (B) : (A))
//#define LIBOCAS_MAX(A,B) ((A) < (B) ? (B) : (A))
#define LIBOCAS_ABS(A) ((A) < 0 ? -(A) : (A))


int rootrmf(double A0, double B0,
        double *Bi, double *Ci, double *lambda,
        int nData
        );

#endif // METHODS_H

