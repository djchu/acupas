#include <stdint.h>
#include "sparse_mat.h"
#include "libocas.h"
#include "ocas_helper.h"
#include "evaluate_multiclass.h"
/*----------------------------------------------------------------------
 It computes dp = full_vec'*sparse_mat(:,col)
 where full_vec is a double array and sparse_mat is Matlab
 sparse matrix.
  ----------------------------------------------------------------------*/


//wtxi = W(:,j)'*X(:,i);
double wtxi_mul(double *full_vec, int j, mxArray *sparse_mat, uint32_t col)
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
                dp += full_vec[j*nDim+row]*val;
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
  evaluation function for multiclass SVM:

  output = argmax_y(data_X'*W^y);
  ----------------------------------------------------------------------*/
double evaluate_multiclass(mxArray *tstX, double *tsty, int nData_tst)
{
    int pred_label = 0, correct = 0;
    double dec_value, max_dfce;
    uint32_t i, j;

    for(i=0; i < nData_tst; i++)
    {
        max_dfce = LIBOCAS_MINUS_INF;
        for(j=0; j<nY; j++)
        {
            dec_value = wtxi_mul(W, j, tstX, i);

            if(dec_value>max_dfce)
            {
                max_dfce = dec_value;
                pred_label = j+1;
            }
        }
        if(pred_label == tsty[i])
            correct++;
    }

    return (double)correct/(double)nData_test;
}
