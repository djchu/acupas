#include <stdint.h>
#include "sparse_mat.h"
#include "libocas.h"
#include "ocas_helper.h"
#include "evaluate_testing.h"

/*----------------------------------------------------------------------
 It computes dp = full_vec'*sparse_mat(:,col)
 where full_vec is a double array and sparse_mat is Matlab
 sparse matrix.
  ----------------------------------------------------------------------*/

double wtxi_bin(double *full_vec, mxArray *sparse_mat, uint32_t col)
{
    uint32_t nItems, ptr, i, row;
    INDEX_TYPE_T *Ir, *Jc;
    double *Pr, val, dp;

    dp = 0;
    if(mxIsSparse(test_X)){
        Ir = mxGetIr(sparse_mat);
        Jc = mxGetJc(sparse_mat);
        Pr = mxGetPr(sparse_mat);

        nItems = Jc[col+1] - Jc[col];
        ptr = Jc[col];

        for(i=0; i < nItems; i++) {
            val = Pr[ptr];
            row = Ir[ptr++];

            if(row<=nDim)
                dp += full_vec[row]*val;
        }
    }
    else{
        double *ptr = mxGetPr(test_X);

        for(row=0; row < nDim; row++ ) {
            val = ptr[LIBOCAS_INDEX(row,col,nDim)];
            dp += full_vec[row]*val;
        }
    }


    return dp;
}

/*----------------------------------------------------------------------
  evaluate testing function:

  output = data_X'*W;
  ----------------------------------------------------------------------*/
double evaluate_testing(mxArray *tstX, double *tsty, int nData_tst)
{
    int correct = 0;
    uint32_t i;

    for(i=0; i < nData_tst; i++) {
        double tmp = tsty[i]*X0*W0;
        double dec_value=0.0;
        //dec_value = dp_sparse_col(W, test_X, i);
        dec_value = wtxi_bin(W, tstX, i);
        dec_value += tmp;
        if(dec_value>0 && tsty[i]==1)
            ++correct;
        else if(dec_value<=0 && tsty[i]==-1)
            ++correct;
    }

    return (double)correct/(double)nData_tst;
}
