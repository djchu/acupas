#ifndef EVULATE_TESTING_H
#define EVULATE_TESTING_H

#include "sparse_mat.h"

// evaluate testing accuracy
double evaluate_testing(mxArray *tstX, double *tsty, int nData_tst);

#endif // EVULATE_TESTING_H

