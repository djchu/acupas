/*-----------------------------------------------------------------------
 * libocas.c: Implementation of the ACUPA and OCAS solvers for training
 *            linear SVM classifiers.
 *
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation;
 *-------------------------------------------------------------------- */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include <stdio.h>
#include <stdint.h>

#include "libocas.h"
#include "libqp.h"

#include "evaluate_testing.h"
#include "evaluate_multiclass.h"
#include "sparse_mat.h"
#include "ocas_helper.h"
#include "features_double.h"

#define epsilon 1e-15
#define MU 0.1

static const uint32_t QPSolverMaxIter = 10000000;

static double *H;
static uint32_t BufSize;

/*----------------------------------------------------------------------
 Returns pointer at i-th column of Hessian matrix.
  ----------------------------------------------------------------------*/
static const double *get_col( uint32_t i)
{
    return( &H[ BufSize*i ] );
}

/*----------------------------------------------------------------------
  Returns time of the day in seconds.
  ----------------------------------------------------------------------
static double get_time()
{
    struct timeval tv;
    if (gettimeofday(&tv, NULL)==0)
        return tv.tv_sec+((double)(tv.tv_usec))/1e6;
    else
        return 0.0;
}
*/


/*----------------------------------------------------------------------
  Linear binary Ocas-SVM solver.
  ----------------------------------------------------------------------*/
ocas_return_value_T svm_ocas_solver(
        double C,
        uint32_t nData,
        double TolRel,
        double TolAbs,
        double QPBound,
        double MaxTime,
        uint32_t _BufSize,
        uint8_t Method,
        void (*compute_W)(double*, double*, double*, uint32_t, void*),
        double (*update_W)(double, void*),
        int (*add_new_cut)(double*, uint32_t*, uint32_t, uint32_t, void*),
        int (*compute_output)(double*, void* ),
        int (*sort)(double*, double*, uint32_t),
        void (*ocas_print)(ocas_return_value_T),
        void* user_data)
{
    ocas_return_value_T ocas={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    double *b, *alpha, *diag_H;
    double *output, *old_output;

    double xi, sq_norm_W, QPSolverTolRel, dot_prod_WoldW, sq_norm_oldW;
    double A0, B0, GradVal, t, t1, t2, *Ci, *Bi, *hpf, *hpb;  // t: stepsize
    double start_time, ocas_start_time;
    double LS_time=0, LS_start_time; // save the linesearch time
    uint32_t num_hp, s; // s: LS iter-num, num_hp: [-Ci/Bi], s.t.-Ci/Bi>0
    double Accuracy;

    FILE *fpresult, *fpinx, *fpbici, *fpbi;
    int DEBUG = 0; // for debug

    uint32_t *U, *G, *L; //index set for randomized median finding
    // double *hpc, *lambi;  //lam_k = lambi[k], hpc is not needed.
    uint32_t j;
    double lam_k;
    U = NULL;
    G = NULL;
    L = NULL;

    if(Method==1) // OCAS: Method == 1
        fpresult = fopen("ocas.plot","w"); // save result to file for plotting with matlab
    else{ //Acupa: Method == 2
        fpresult = fopen("acupa.plot","w");

        lam_k=0;
        U = (uint32_t*) LIBOCAS_CALLOC(nData, sizeof(uint32_t));
        G = (uint32_t*) LIBOCAS_CALLOC(nData, sizeof(uint32_t));
        L = (uint32_t*) LIBOCAS_CALLOC(nData, sizeof(uint32_t));
    }

    uint32_t cut_length;
    uint32_t i, *new_cut;
    uint32_t *I;
    uint8_t S = 1;
    libqp_state_T qp_exitflag;

    ocas_start_time = get_time();
    //ocas.qp_solver_time = 0;
    //ocas.output_time = 0;
    //ocas.sort_time = 0;
    //ocas.add_time = 0;
    //ocas.w_time = 0;
    //ocas.print_time = 0;

    BufSize = _BufSize;

    QPSolverTolRel = TolRel*0.5;

    H=NULL;
    b=NULL;
    alpha=NULL;
    new_cut=NULL;
    I=NULL;
    diag_H=NULL;
    output=NULL;
    old_output=NULL;
    hpf=NULL;
    hpb = NULL;
    Ci=NULL;
    Bi=NULL;

    /* Hessian matrix contains dot product of normal vectors of selected cutting planes */
    H = (double*)LIBOCAS_CALLOC(BufSize*BufSize,sizeof(double));
    if(H == NULL)
    {
        ocas.exitflag=-2;
        goto cleanup;
    }

    /* bias of cutting planes */
    b = (double*)LIBOCAS_CALLOC(BufSize,sizeof(double));
    if(b == NULL)
    {
        ocas.exitflag=-2;
        goto cleanup;
    }

    alpha = (double*)LIBOCAS_CALLOC(BufSize,sizeof(double));
    if(alpha == NULL)
    {
        ocas.exitflag=-2;
        goto cleanup;
    }

    /* indices of examples which define a new cut */
    new_cut = (uint32_t*)LIBOCAS_CALLOC(nData,sizeof(uint32_t));
    if(new_cut == NULL)
    {
        ocas.exitflag=-2;
        goto cleanup;
    }

    I = (uint32_t*)LIBOCAS_CALLOC(BufSize,sizeof(uint32_t));
    if(I == NULL)
    {
        ocas.exitflag=-2;
        goto cleanup;
    }

    for(i=0; i< BufSize; i++) I[i] = 1;

    diag_H = (double*)LIBOCAS_CALLOC(BufSize,sizeof(double));
    if(diag_H == NULL)
    {
        ocas.exitflag=-2;
        goto cleanup;
    }

    output = (double*)LIBOCAS_CALLOC(nData,sizeof(double));
    if(output == NULL)
    {
        ocas.exitflag=-2;
        goto cleanup;
    }

    old_output = (double*)LIBOCAS_CALLOC(nData,sizeof(double));
    if(old_output == NULL)
    {
        ocas.exitflag=-2;
        goto cleanup;
    }

    /* array of hinge points used in line-serach  */
    hpf = (double*) LIBOCAS_CALLOC(nData, sizeof(hpf[0]));
    if(hpf == NULL)
    {
        ocas.exitflag=-2;
        goto cleanup;
    }

    hpb = (double*) LIBOCAS_CALLOC(nData, sizeof(hpb[0]));
    if(hpb == NULL)
    {
        ocas.exitflag=-2;
        goto cleanup;
    }

    /* vectors Ci, Bi are used in the line search procedure */
    Ci = (double*)LIBOCAS_CALLOC(nData,sizeof(double));
    if(Ci == NULL)
    {
        ocas.exitflag=-2;
        goto cleanup;
    }

    Bi = (double*)LIBOCAS_CALLOC(nData,sizeof(double));
    if(Bi == NULL)
    {
        ocas.exitflag=-2;
        goto cleanup;
    }

    ocas.nCutPlanes = 0;
    ocas.exitflag = 0;
    ocas.nIter = 0;

    /* Compute initial value of Q_P assuming that W is zero vector.*/
    sq_norm_W = 0;
    xi = nData;
    ocas.Q_P = 0.5*sq_norm_W + C*xi;
    ocas.Q_D = 0;

    /* Compute the initial cutting plane */
    cut_length = nData;
    for(i=0; i < nData; i++)
        new_cut[i] = i;

    ocas.trn_err = nData;
    ocas.ocas_time = get_time() - ocas_start_time;

    /*  ocas_print("%4d: tim=%f, Q_P=%f, Q_D=%f, Q_P-Q_D=%f, Q_P-Q_D/abs(Q_P)=%f\n",
          ocas.nIter,cur_time, ocas.Q_P,ocas.Q_D,ocas.Q_P-ocas.Q_D,(ocas.Q_P-ocas.Q_D)/LIBOCAS_ABS(ocas.Q_P));
  */
    ocas_print(ocas);

    /* main loop */
    while( ocas.exitflag == 0 )
    {
        start_time = get_time();

        ocas.nIter++;

        /* append a new cut to the buffer and update H */
        b[ocas.nCutPlanes] = -(double)cut_length;

        if(add_new_cut( &H[LIBOCAS_INDEX(0,ocas.nCutPlanes,BufSize)], new_cut, cut_length, ocas.nCutPlanes, user_data ) != 0)
        {
            ocas.exitflag=-2;
            goto cleanup;
        }

        //ocas.add_time += get_time() - start_time;

        /* copy new added row:  H(ocas.nCutPlanes,ocas.nCutPlanes,1:ocas.nCutPlanes-1) = H(1:ocas.nCutPlanes-1:ocas.nCutPlanes)' */
        diag_H[ocas.nCutPlanes] = H[LIBOCAS_INDEX(ocas.nCutPlanes,ocas.nCutPlanes,BufSize)];
        for(i=0; i < ocas.nCutPlanes; i++) {
            H[LIBOCAS_INDEX(ocas.nCutPlanes,i,BufSize)] = H[LIBOCAS_INDEX(i,ocas.nCutPlanes,BufSize)];
        }

        ocas.nCutPlanes++;

        /* call inner QP solver */
        //start_time = get_time();

        qp_exitflag = libqp_splx_solver(&get_col, diag_H, b, &C, I, &S, alpha,
                                        ocas.nCutPlanes, QPSolverMaxIter, 0.0, QPSolverTolRel, -LIBOCAS_PLUS_INF,0);

        ocas.qp_exitflag = qp_exitflag.exitflag;

        //ocas.qp_solver_time += get_time() - start_time;
        ocas.Q_D = -qp_exitflag.QP;

        ocas.nNZAlpha = 0;
        for(i=0; i < ocas.nCutPlanes; i++) {
            if( alpha[i] != 0) ocas.nNZAlpha++;
        }

        sq_norm_oldW = sq_norm_W;
        //start_time = get_time();
        compute_W( &sq_norm_W, &dot_prod_WoldW, alpha, ocas.nCutPlanes, user_data );
        //ocas.w_time += get_time() - start_time;

        /* select a new cut */
        switch( Method )
        {
        /* cutting plane algorithm implemented in SVMperf and BMRM */
        case 0:

            //start_time = get_time();
            if( compute_output( output, user_data ) != 0)
            {
                ocas.exitflag=-2;
                goto cleanup;
            }
            //ocas.output_time += get_time()-start_time;

            xi = 0;
            cut_length = 0;
            ocas.trn_err = 0;
            for(i=0; i < nData; i++)
            {
                if(output[i] <= 0) ocas.trn_err++;

                if(output[i] <= 1) {
                    xi += 1 - output[i];
                    new_cut[cut_length] = i;
                    cut_length++;
                }
            }
            ocas.Q_P = 0.5*sq_norm_W + C*xi;

            ocas.ocas_time += get_time() - start_time;

            /*        ocas_print("%4d: tim=%f, Q_P=%f, Q_D=%f, Q_P-Q_D=%f, 1-Q_D/Q_P=%f, nza=%4d, err=%.2f%%, qpf=%d\n",
                  ocas.nIter,cur_time, ocas.Q_P,ocas.Q_D,ocas.Q_P-ocas.Q_D,(ocas.Q_P-ocas.Q_D)/LIBOCAS_ABS(ocas.Q_P),
                  ocas.nNZAlpha, 100*(double)ocas.trn_err/(double)nData, ocas.qp_exitflag );
        */

            //start_time = get_time();
            ocas_print(ocas);
            //ocas.print_time += get_time() - start_time;

            break;



        case 2: // Acupa strategy
            if(DEBUG){
                printf("acupaIter:%d\n",ocas.nIter);
                fpinx = fopen("indexfile", "w"); // all random index in U saved in inxfile
                fpbici = fopen("bicifile","w"); //save all bi and ci into file
                fpbi = fopen("bifile","w");  //save all bi when compute deriv. at 0
            }
            LS_start_time = get_time();

            A0 = sq_norm_W -2*dot_prod_WoldW + sq_norm_oldW;
            B0 = dot_prod_WoldW - sq_norm_oldW;

            memcpy( old_output, output, sizeof(double)*nData );

            if( compute_output( output, user_data ) != 0)
            {
                ocas.exitflag=-2;
                goto cleanup;
            }

            num_hp = 0;
            GradVal = B0;

            for(i=0; i< nData; i++)
            {  // Choose Bi and Ci, and compute deriv. at 0.
                Ci[i] = C*(1-old_output[i]);
                Bi[i] = C*(old_output[i] - output[i]);
                if(DEBUG)
                    fprintf(fpbici,"%.10f\t%.10f\n", Bi[i], Ci[i]);

                double val;
                if(Bi[i] != 0)
                    val = -Ci[i]/Bi[i];
                else
                    val = -LIBOCAS_PLUS_INF;

                if (val>0)
                {
                    U[num_hp] = num_hp;  //RMF: U=[n]
                    hpb[num_hp] = Bi[i];
                    hpf[num_hp] = val; //lambi[num_hp] = val;

                    num_hp++;
                }

                //                if( (Bi[i] < 0 && val > 0) || (Bi[i] > 0 && val <= 0))
                //                    GradVal += Bi[i];
                if (Ci[i]>=0){
                    if((Ci[i]==0) & (Bi[i]<0))
                        continue;
                    GradVal += Bi[i];
                    //fprintf(fpbi,"%.10f ",Bi[i]);
                }
            }
            if(DEBUG){
                fclose(fpbi);
                fclose(fpbici);
            }

            // Acupa line-search begins
            t = 0;
            if(GradVal < 0){
                uint32_t inx, k=0, k_old, Ulength = num_hp;
                double lam_old;
                lam_k = 0;

                //srand(time(NULL));
                for(s=0; s<num_hp; s++)
                {
                    inx = rand() % Ulength; // inx \in [0,...[U]-1]

                    k_old = k;
                    k = U[inx];  // Random choose k \in U.
                    lam_old = lam_k;
                    lam_k = hpf[k]; //lam_k = lambi[k];

                    if(DEBUG)
                        fprintf(fpinx,"%d\t%d\t%lf\t%d\n",
                                s, Ulength, lam_k, inx);  // save index in file

                    int gi=0, li=0;
                    for(i=0; i<Ulength; i++){ // set G and L
                        double lambiUi = hpf[U[i]];  // double lamiUi = lambi[U[i]];
                        if(lambiUi >= lam_k) {
                            G[gi] = U[i];
                            gi++;  // gi is the number of G
                        }
                        else {
                            L[li] = U[i];
                            li++;  // li is the number of L
                        }
                    }

                    if(lam_k >= lam_old){
                        GradVal +=  A0*(lam_k-lam_old);
                        for(i=0; i<li; i++)
                            GradVal += fabs(hpb[L[i]]);
                    }
                    else{
                        GradVal -= (fabs(hpb[k_old]) + A0*(lam_old-lam_k));
                        for(i=0; i<gi; i++)
                            GradVal -= fabs(hpb[G[i]]);
                    }

                    if(GradVal<0){
                        GradVal += LIBOCAS_ABS(hpb[k]); // get sup g(lam_k)
                        if(GradVal>=0)
                            break;
                        else{
                            for(i=0,j=0; i<gi; i++){
                                if(G[i]==k)
                                    continue;
                                U[j] = G[i];
                                j++;
                            }
                            Ulength = gi-1;
                        }
                    }
                    else{ // GradVal>=0
                        if(li>0)
                            GradVal += LIBOCAS_ABS(hpb[k]);
                        for(i=0; i<li; i++)
                            U[i] = L[i];
                        Ulength = li;
                    }

                    if(Ulength==0){
                        lam_k -= GradVal/A0;
                        break;
                    }
                }
                if(DEBUG){
                    fclose(fpinx);
                }
                t = lam_k;
            }
            // Acupa line-search ends

            LS_time += get_time() - LS_start_time;

            t = LIBOCAS_MAX(t,0);          /* just sanity check; t < 0 should not ocure */

            t1 = t;                /* new (best so far) W */
            t2 = t+MU*(1.0-t);   /* new cutting plane */

            /* update W to be the best so far solution */
            sq_norm_W = update_W( t1, user_data );

            /* select a new cut */
            xi = 0;
            cut_length = 0;
            ocas.trn_err = 0;
            for(i=0; i < nData; i++ ) {

                if( (old_output[i]*(1-t2) + t2*output[i]) <= 1 )
                {
                    new_cut[cut_length] = i;
                    cut_length++;
                }

                output[i] = old_output[i]*(1-t1) + t1*output[i];

                if( output[i] <= 1) xi += 1-output[i];
                if( output[i] <= 0) ocas.trn_err++;

            }

            ocas.Q_P = 0.5*sq_norm_W + C*xi;

            ocas.ocas_time += get_time() - LS_start_time;

            ocas_print(ocas);

            Accuracy = evaluate_testing(test_X, test_y, nData_test);

            printf("iter:%d time:%lf pf:%lf df:%lf acc:%lf lstime:%lf\n",
                   ocas.nIter, ocas.ocas_time, ocas.Q_P, ocas.Q_D, Accuracy, LS_time);
            fprintf(fpresult,"%d\t%lf\t%lf\t%lf\t%lf\t%lf\n",
                    ocas.nIter,ocas.ocas_time, ocas.Q_P, ocas.Q_D, Accuracy, LS_time);

            break;

            /* Ocas strategy */
        case 1:

            LS_start_time = get_time();
            /* Linesearch */
            A0 = sq_norm_W -2*dot_prod_WoldW + sq_norm_oldW;
            B0 = dot_prod_WoldW - sq_norm_oldW;

            memcpy( old_output, output, sizeof(double)*nData );

            //start_time = get_time();
            if( compute_output( output, user_data ) != 0)
            {
                ocas.exitflag=-2;
                goto cleanup;
            }
            ocas.output_time += get_time()-start_time;

            //uint32_t num_hp = 0;
            num_hp = 0;
            GradVal = B0;

            for(i=0; i< nData; i++) {

                Ci[i] = C*(1-old_output[i]);
                Bi[i] = C*(old_output[i] - output[i]);

                double val;
                if(Bi[i] != 0)
                    val = -Ci[i]/Bi[i];
                else
                    val = -LIBOCAS_PLUS_INF;

                if (val>0)
                {
                    /*            hpi[num_hp] = i;*/
                    hpb[num_hp] = Bi[i];
                    hpf[num_hp] = val;
                    num_hp++;
                }

                if( (Bi[i] < 0 && val > 0) || (Bi[i] > 0 && val <= 0))
                    GradVal += Bi[i];

            }

            t = 0;
            if( GradVal < 0 )
            {
                //start_time = get_time();
                /*          if( sort(hpf, hpi, num_hp) != 0)*/
                if( sort(hpf, hpb, num_hp) != 0 )
                {
                    ocas.exitflag=-2;
                    goto cleanup;
                }
                //ocas.sort_time += get_time() - start_time;

                double t_new, GradVal_new;
                i = 0;
                while( GradVal < 0 && i < num_hp )
                {
                    t_new = hpf[i];

                    GradVal_new = GradVal + LIBOCAS_ABS(hpb[i]) + A0*(t_new-t);
                    if( GradVal_new >= 0 )
                    {
                        t = t + GradVal*(t-t_new)/(GradVal_new - GradVal);
                    }
                    else
                    {
                        t = t_new;
                        i++;
                    }

                    //                    GradVal_new = GradVal + A0*(t_new-t);
                    //                    if( GradVal_new >=0 )
                    //                        t = t + GradVal*(t-t_new)/(GradVal_new - GradVal);
                    //                    else{
                    //                        t = t_new;
                    //                        GradVal_new = GradVal_new + LIBOCAS_ABS(hpb[i]);
                    //                        i++;
                    //                    }

                    GradVal = GradVal_new;
                }
            }
            s = i;
            LS_time += get_time() - LS_start_time;

            t = LIBOCAS_MAX(t,0);          /* just sanity check; t < 0 should not ocure */

            t1 = t;                /* new (best so far) W */
            t2 = t+MU*(1.0-t);   /* new cutting plane */
            /*        t2 = t+(1.0-t)/10.0;   */

            /* update W to be the best so far solution */
            sq_norm_W = update_W( t1, user_data );

            /* select a new cut */
            xi = 0;
            cut_length = 0;
            ocas.trn_err = 0;
            for(i=0; i < nData; i++ ) {

                if( (old_output[i]*(1-t2) + t2*output[i]) <= 1 )
                {
                    new_cut[cut_length] = i;
                    cut_length++;
                }

                output[i] = old_output[i]*(1-t1) + t1*output[i];

                if( output[i] <= 1) xi += 1-output[i];
                if( output[i] <= 0) ocas.trn_err++;

            }

            ocas.Q_P = 0.5*sq_norm_W + C*xi;

            ocas.ocas_time += get_time() - LS_start_time;

            /*        ocas_print("%4d: tim=%f, Q_P=%f, Q_D=%f, Q_P-Q_D=%f, 1-Q_D/Q_P=%f, nza=%4d, err=%.2f%%, qpf=%d\n",
                   ocas.nIter, cur_time, ocas.Q_P,ocas.Q_D,ocas.Q_P-ocas.Q_D,(ocas.Q_P-ocas.Q_D)/LIBOCAS_ABS(ocas.Q_P),
                   ocas.nNZAlpha, 100*(double)ocas.trn_err/(double)nData, ocas.qp_exitflag );
        */

            //start_time = get_time();

            //ocas_print(ocas);

            //Accuracy = evaluate_testing();
            Accuracy = evaluate_testing(test_X, test_y, nData_test);

            //ocas.print_time += get_time() - start_time;

            printf("iter:%d time:%lf pf:%lf df:%lf acc:%lf lstime:%lf\n",
                   ocas.nIter, ocas.ocas_time, ocas.Q_P, ocas.Q_D, Accuracy, LS_time);
            fprintf(fpresult,"%d\t%lf\t%lf\t%lf\t%lf\t%lf\n",
                    ocas.nIter,ocas.ocas_time, ocas.Q_P, ocas.Q_D, Accuracy, LS_time);

            break;
        }

        /* Stopping conditions */
        if( ocas.Q_P - ocas.Q_D <= TolRel*LIBOCAS_ABS(ocas.Q_P)) ocas.exitflag = 1;
        if( ocas.Q_P - ocas.Q_D <= TolAbs) ocas.exitflag = 2;
        if( ocas.Q_P <= QPBound) ocas.exitflag = 3;
        if( ocas.ocas_time >= MaxTime) ocas.exitflag = 4;
        if(ocas.nCutPlanes >= BufSize) ocas.exitflag = -1;

    } /* end of the main loop */

    fclose(fpresult);

cleanup:

    LIBOCAS_FREE(H);
    LIBOCAS_FREE(b);
    LIBOCAS_FREE(alpha);
    LIBOCAS_FREE(new_cut);
    LIBOCAS_FREE(I);
    LIBOCAS_FREE(diag_H);
    LIBOCAS_FREE(output);
    LIBOCAS_FREE(old_output);
    LIBOCAS_FREE(hpf);
    /*  LIBOCAS_FREE(hpi);*/
    LIBOCAS_FREE(hpb);
    LIBOCAS_FREE(Ci);
    LIBOCAS_FREE(Bi);

    if(Method==2){
        LIBOCAS_FREE(U);
        LIBOCAS_FREE(G);
        LIBOCAS_FREE(L);
    }
    ocas.ocas_time = get_time() - ocas_start_time;

    return(ocas);
}


/*----------------------------------------------------------------------
  Multiclass SVM-Ocas solver
  ----------------------------------------------------------------------*/

/* Helper function needed by the multi-class SVM linesearch.

  - This function finds a simplified representation of a piece-wise linear function
  by splitting the domain into intervals and fining active terms for these intevals */
static void findactive(double *Theta, double *SortedA, uint32_t *nSortedA, double *A, double *B, int n,
                       int (*sort)(double*, double*, uint32_t))
{
    double tmp, theta;
    uint32_t i, j, idx, idx2 = 0, start;

    sort(A,B,n);

    tmp = B[0];
    idx = 0;
    i = 0;
    while( i < (uint32_t)n-1 && A[i] == A[i+1])
    {
        if( B[i+1] > B[idx] )
        {
            idx = i+1;
            tmp = B[i+1];
        }
        i++;
    }

    (*nSortedA) = 1;
    SortedA[0] = A[idx];

    while(1)
    {
        start = idx + 1;
        while( start < (uint32_t)n && A[idx] == A[start])
            start++;

        theta = LIBOCAS_PLUS_INF;
        for(j=start; j < (uint32_t)n; j++)
        {
            tmp = (B[j] - B[idx])/(A[idx]-A[j]);
            if( tmp < theta)
            {
                theta = tmp;
                idx2 = j;
            }
        }

        if( theta < LIBOCAS_PLUS_INF)
        {
            Theta[(*nSortedA) - 1] = theta;
            SortedA[(*nSortedA)] = A[idx2];
            (*nSortedA)++;
            idx = idx2;
        }
        else
            return;
    }
}

/*
 * This function finds a simplified representation of
 * a piece-wise linear(PWL) function.
*/
static void findpwl(double *S, double *Lambda, int *count,
                    double *B, double *C, int n, double *grad0)
{
    double btmp, ctmp, tmp, lamR = LIBOCAS_MINUS_INF, lamR2;
    uint32_t i, j, k, jR = 0, jR2;
    uint32_t YhatSet[n], numYhatset = 0;

    btmp = LIBOCAS_MINUS_INF;
    ctmp = C[0];
    for(i = 0; i <= n-1; i++) // compute sup g(0)
        if(C[i] > ctmp){
            ctmp = C[i];
            btmp = B[i];
        }
        else
            if((C[i]==ctmp) & (B[i]>btmp)){
                ctmp = C[i];
                btmp = B[i];
            }
    *grad0 += btmp;

    btmp = B[0];
    ctmp = LIBOCAS_MINUS_INF;
    for(i = 0; i <= n-1; i++) // find jR
        if(B[i] > btmp){
            btmp = B[i];
            jR = i;
        }
        else
            if((B[i]==btmp) & (C[i]>ctmp))
                jR = i;

    for(j=0, i=0; j<=n-1; j++){
        if(B[j]<B[jR]){
            YhatSet[i] = j;
            i++;
        }
    }
    numYhatset = i;

    while(numYhatset>0){
        jR2 = jR;
        lamR2 = lamR;

        lamR = LIBOCAS_MINUS_INF;
        for(i=0; i<numYhatset; i++){
            j = YhatSet[i];
            tmp = (C[j]-C[jR2])/(B[jR2]-B[j]);
            if(tmp>lamR){
                lamR = tmp;
                jR = j;
            }
        }
        if(lamR<=0)
            break;

        for(i=0, k=0; i<numYhatset; i++){
            j = YhatSet[i];
            if(B[j]<B[jR]){
                YhatSet[k] = j;
                k++;
            }
        }
        numYhatset = k;

        if(lamR==lamR2)
            continue;

        S[*count] = LIBOCAS_ABS(B[jR2]-B[jR]);
        Lambda[*count] = lamR;
        (*count)++;
    }

}

/*----------------------------------------------------------------------
  Multiclass linear OCAS-SVM solver.
  ----------------------------------------------------------------------*/
ocas_return_value_T msvm_ocas_solver(
        double C,
        double *data_y,
        uint32_t nY,
        uint32_t nData,
        double TolRel,
        double TolAbs,
        double QPBound,
        double MaxTime,
        uint32_t _BufSize,
        uint8_t Method,
        void (*compute_W)(double*, double*, double*, uint32_t, void*),
        double (*update_W)(double, void*),
        int (*add_new_cut)(double*, uint32_t*, uint32_t, void*),
        int (*compute_output)(double*, void* ),
        int (*sort)(double*, double*, uint32_t),
        void (*ocas_print)(ocas_return_value_T),
        void* user_data)
{
    ocas_return_value_T ocas={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    double *b, *alpha, *diag_H;
    double *output, *old_output;
    double xi, sq_norm_W, QPSolverTolRel, QPSolverTolAbs, dot_prod_WoldW, sq_norm_oldW;
    double A0, B0, t, t1, t2, R, tmp, element_b, x;
    double *A, *B, *theta, *Theta, *sortedA, *Add;
    double start_time, ocas_start_time, grad_sum, grad, min_x = 0, old_x, old_grad;
    uint32_t i, y, y2, ypred = 0, *new_cut, cnt1, cnt2, j, nSortedA, idx;
    uint32_t *I;
    uint8_t S = 1;
    libqp_state_T qp_exitflag;

    // my variables:
    int DEBUG = 0; // flag to debug

    // all the iteration time was saved in ocas.ocas_time
    double LS_time = 0.0, LS_start_time; // save the linesearch time
    double PWL_time = 0.0, PWL_start_time; // save the active PWL functions time
    int numLambda, s; // s: LS iter-num; numLambda: [\lambda>0]
    double lam_k, GradVal, Accuracy=0.123456;

    FILE *fpresult=NULL, *fpinx;  //save result in file
    uint32_t *U, *G, *L;  // index set for randomized median finding
    U = NULL;
    G = NULL;
    L = NULL;

    if(Method==1) // OCAS: Method == 1
        fpresult = fopen("ocaM.plot","w"); // save result to file for plotting with matlab
    else
        if(Method == 2)
        { //Acupa: Method == 2
            fpresult = fopen("acupaM.plot","w");

            U = (uint32_t*) LIBOCAS_CALLOC(nData*nY, sizeof(uint32_t));
            G = (uint32_t*) LIBOCAS_CALLOC(nData*nY, sizeof(uint32_t));
            L = (uint32_t*) LIBOCAS_CALLOC(nData*nY, sizeof(uint32_t));
        }


    ocas_start_time = get_time();
    //ocas.qp_solver_time = 0;
    //ocas.output_time = 0;
    //ocas.sort_time = 0;
    //ocas.add_time = 0;
    //ocas.w_time = 0;
    //ocas.print_time = 0;

    BufSize = _BufSize;

    QPSolverTolRel = TolRel*0.5;
    QPSolverTolAbs = TolAbs*0.5;

    H=NULL;
    b=NULL;
    alpha=NULL;
    new_cut=NULL;
    I=NULL;
    diag_H=NULL;
    output=NULL;
    old_output=NULL;
    A = NULL;
    B = NULL;
    theta = NULL;
    Theta = NULL;
    sortedA = NULL;
    Add = NULL;

    /* Hessian matrix contains dot product of normal vectors of selected cutting planes */
    H = (double*)LIBOCAS_CALLOC(BufSize*BufSize,sizeof(double));
    if(H == NULL)
    {
        ocas.exitflag=-2;
        goto cleanup;
    }

    /* bias of cutting planes */
    b = (double*)LIBOCAS_CALLOC(BufSize,sizeof(double));
    if(b == NULL)
    {
        ocas.exitflag=-2;
        goto cleanup;
    }

    alpha = (double*)LIBOCAS_CALLOC(BufSize,sizeof(double));
    if(alpha == NULL)
    {
        ocas.exitflag=-2;
        goto cleanup;
    }

    /* indices of examples which define a new cut */
    new_cut = (uint32_t*)LIBOCAS_CALLOC(nData,sizeof(uint32_t));
    if(new_cut == NULL)
    {
        ocas.exitflag=-2;
        goto cleanup;
    }

    I = (uint32_t*)LIBOCAS_CALLOC(BufSize,sizeof(uint32_t));
    if(I == NULL)
    {
        ocas.exitflag=-2;
        goto cleanup;
    }

    for(i=0; i< BufSize; i++)
        I[i] = 1;

    diag_H = (double*)LIBOCAS_CALLOC(BufSize,sizeof(double));
    if(diag_H == NULL)
    {
        ocas.exitflag=-2;
        goto cleanup;
    }

    output = (double*)LIBOCAS_CALLOC(nData*nY,sizeof(double));
    if(output == NULL)
    {
        ocas.exitflag=-2;
        goto cleanup;
    }

    old_output = (double*)LIBOCAS_CALLOC(nData*nY,sizeof(double));
    if(old_output == NULL)
    {
        ocas.exitflag=-2;
        goto cleanup;
    }

    /* auxciliary variables used in the linesearch */
    A = (double*)LIBOCAS_CALLOC(nData*nY,sizeof(double));
    if(A == NULL)
    {
        ocas.exitflag=-2;
        goto cleanup;
    }

    B = (double*)LIBOCAS_CALLOC(nData*nY,sizeof(double));
    if(B == NULL)
    {
        ocas.exitflag=-2;
        goto cleanup;
    }

    theta = (double*)LIBOCAS_CALLOC(nY,sizeof(double));
    if(theta == NULL)
    {
        ocas.exitflag=-2;
        goto cleanup;
    }

    sortedA = (double*)LIBOCAS_CALLOC(nY,sizeof(double));
    if(sortedA == NULL)
    {
        ocas.exitflag=-2;
        goto cleanup;
    }

    Theta = (double*)LIBOCAS_CALLOC(nData*nY,sizeof(double));
    if(Theta == NULL)
    {
        ocas.exitflag=-2;
        goto cleanup;
    }

    Add = (double*)LIBOCAS_CALLOC(nData*nY,sizeof(double));
    if(Add == NULL)
    {
        ocas.exitflag=-2;
        goto cleanup;
    }

    /* Set initial values*/
    ocas.nCutPlanes = 0;
    ocas.exitflag = 0;
    ocas.nIter = 0;
    ocas.Q_D = 0;
    ocas.trn_err = nData;
    R = (double)nData;
    sq_norm_W = 0;
    element_b = (double)nData;
    ocas.Q_P = 0.5*sq_norm_W + C*R;

    /* initial cutting plane */
    for(i=0; i < nData; i++)
    {
        y2 = (uint32_t)data_y[i]-1;

        if(y2 > 0)
            new_cut[i] = 0;  // examples not in class 1
        else
            new_cut[i] = 1;  // examples in class 1

    }

    ocas.ocas_time = get_time() - ocas_start_time;

    //ocas_print(ocas);

    /* main loop of the OCAS */
    while( ocas.exitflag == 0 )
    {
        start_time = get_time();

        ocas.nIter++;

        /* append a new cut to the buffer and update H */
        b[ocas.nCutPlanes] = -(double)element_b;

        if(add_new_cut( &H[LIBOCAS_INDEX(0,ocas.nCutPlanes,BufSize)], new_cut, ocas.nCutPlanes, user_data ) != 0)
        {
            ocas.exitflag=-2;
            goto cleanup;
        }

        /* copy newly appended row: H(ocas.nCutPlanes,ocas.nCutPlanes,1:ocas.nCutPlanes-1) = H(1:ocas.nCutPlanes-1:ocas.nCutPlanes)' */
        diag_H[ocas.nCutPlanes] = H[LIBOCAS_INDEX(ocas.nCutPlanes,ocas.nCutPlanes,BufSize)]; // diagonal matrix of H
        for(i=0; i < ocas.nCutPlanes; i++)
        {
            H[LIBOCAS_INDEX(ocas.nCutPlanes,i,BufSize)] = H[LIBOCAS_INDEX(i,ocas.nCutPlanes,BufSize)];
        }

        ocas.nCutPlanes++;

        qp_exitflag = libqp_splx_solver(&get_col, diag_H, b, &C, I, &S, alpha,
                                        ocas.nCutPlanes, QPSolverMaxIter, QPSolverTolAbs, QPSolverTolRel, -LIBOCAS_PLUS_INF,0);

        ocas.qp_exitflag = qp_exitflag.exitflag;

        ocas.qp_solver_time += get_time() - start_time;
        ocas.Q_D = -qp_exitflag.QP;

        ocas.nNZAlpha = 0;
        for(i=0; i < ocas.nCutPlanes; i++)
            if( alpha[i] != 0) ocas.nNZAlpha++;

        sq_norm_oldW = sq_norm_W;
        compute_W( &sq_norm_W, &dot_prod_WoldW, alpha, ocas.nCutPlanes, user_data );

        /* select a new cut */
        switch( Method )
        {
        /* cutting plane algorithm implemented in SVMperf and BMRM */
        case 0:

            if( compute_output( output, user_data ) != 0)
            {
                ocas.exitflag=-2;
                goto cleanup;
            }

            /* the following loop computes: */
            element_b = 0.0;    /*  element_b = R(old_W) - g'*old_W */
            R = 0;              /*  R(W) = sum_i max_y ( [[y != y_i]] + (w_y- w_y_i)'*x_i )    */
            ocas.trn_err = 0;   /*  trn_err = sum_i [[y != y_i ]]                              */
            /* new_cut[i] = argmax_i ( [[y != y_i]] + (w_y- w_y_i)'*x_i )  */
            for(i=0; i < nData; i++)
            {
                y2 = (uint32_t)data_y[i]-1;

                for(xi=-LIBOCAS_PLUS_INF, y=0; y < nY; y++)
                {
                    if(y2 != y && xi < output[LIBOCAS_INDEX(y,i,nY)])
                    {
                        xi = output[LIBOCAS_INDEX(y,i,nY)];
                        ypred = y;
                    }
                }

                if(xi >= output[LIBOCAS_INDEX(y2,i,nY)])
                    ocas.trn_err ++;

                xi = LIBOCAS_MAX(0,xi+1-output[LIBOCAS_INDEX(y2,i,nY)]);
                R += xi;
                if(xi > 0)
                {
                    element_b++;
                    new_cut[i] = ypred;
                }
                else
                    new_cut[i] = y2;
            }

            ocas.Q_P = 0.5*sq_norm_W + C*R;

            ocas.ocas_time += get_time() - start_time;

            ocas_print(ocas);
            fflush(stdout);

            break;

            /* The OCAS solver */
        case 1:
            LS_start_time = get_time();
            memcpy( old_output, output, sizeof(double)*nData*nY );

            if( compute_output( output, user_data ) != 0)
            {
                ocas.exitflag=-2;
                goto cleanup;
            }
            ocas.output_time += get_time()-start_time;

            A0 = sq_norm_W - 2*dot_prod_WoldW + sq_norm_oldW;
            B0 = dot_prod_WoldW - sq_norm_oldW;

            for(i=0; i < nData; i++)
            {
                y2 = (uint32_t)data_y[i]-1;

                for(y=0; y < nY; y++)
                {
                    A[LIBOCAS_INDEX(y,i,nY)] = C*(output[LIBOCAS_INDEX(y,i,nY)] - old_output[LIBOCAS_INDEX(y,i,nY)]
                            + old_output[LIBOCAS_INDEX(y2,i,nY)] - output[LIBOCAS_INDEX(y2,i,nY)]);
                    B[LIBOCAS_INDEX(y,i,nY)] = C*(old_output[LIBOCAS_INDEX(y,i,nY)] - old_output[LIBOCAS_INDEX(y2,i,nY)]
                            + (double)(y != y2));
                }
            }

            /* linesearch */
            /*      new_x = msvm_linesearch_mex(A0,B0,AA*C,BB*C);*/

            grad_sum = B0;
            cnt1 = 0;
            cnt2 = 0;

            PWL_start_time = get_time();
            for(i=0; i < nData; i++)
            {
                findactive(theta,sortedA,&nSortedA,&A[i*nY],&B[i*nY],nY,sort);

                idx = 0;
                while( idx < nSortedA-1 && theta[idx] < 0 )
                    idx++;

                grad_sum += sortedA[idx];

                for(j=idx; j < nSortedA-1; j++)
                {
                    Theta[cnt1] = theta[j];
                    cnt1++;
                }

                for(j=idx+1; j < nSortedA; j++)
                {
                    Add[cnt2] = -sortedA[j-1]+sortedA[j];
                    cnt2++;
                }
            }
            PWL_time = get_time() - PWL_start_time;

            sort(Theta,Add,cnt1);

            grad = grad_sum;
            if(grad >= 0)
            {
                min_x = 0;
            }
            else
            {
                old_x = 0;
                old_grad = grad;

                for(i=0; i < cnt1; i++)
                {
                    x = Theta[i];

                    grad = x*A0 + grad_sum;

                    if(grad >=0)
                    {

                        min_x = (grad*old_x - old_grad*x)/(grad - old_grad);

                        break;
                    }
                    else
                    {
                        grad_sum = grad_sum + Add[i];

                        grad = x*A0 + grad_sum;
                        if( grad >= 0)
                        {
                            min_x = x;
                            break;
                        }
                    }

                    old_grad = grad;
                    old_x = x;
                }
            }
            /* end of the linesearch which outputs min_x */
            LS_time += get_time()-LS_start_time;

            t = min_x;
            t1 = t;                /* new (best so far) W */
            t2 = t+(1.0-t)*MU;   /* new cutting plane */
            //t2 = t1;
            /*        t2 = t+(1.0-t)/10.0;    */

            /* update W to be the best so far solution */
            sq_norm_W = update_W( t1, user_data );

            /* the following code  computes a new cutting plane: */
            element_b = 0.0;    /*  element_b = R(old_W) - g'*old_W */
            /* new_cut[i] = argmax_i ( [[y != y_i]] + (w_y- w_y_i)'*x_i )  */
            for(i=0; i < nData; i++)
            {
                y2 = (uint32_t)data_y[i]-1;

                for(xi=-LIBOCAS_PLUS_INF, y=0; y < nY; y++)
                {
                    tmp = old_output[LIBOCAS_INDEX(y,i,nY)]*(1-t2) + t2*output[LIBOCAS_INDEX(y,i,nY)];
                    if(y2 != y && xi < tmp)
                    {
                        xi = tmp;
                        ypred = y;
                    }
                }

                tmp = old_output[LIBOCAS_INDEX(y2,i,nY)]*(1-t2) + t2*output[LIBOCAS_INDEX(y2,i,nY)];
                xi = LIBOCAS_MAX(0,xi+1-tmp);
                if(xi > 0)
                {
                    element_b++;
                    new_cut[i] = ypred;
                }
                else
                    new_cut[i] = y2;
            }

            /* compute Risk, class. error and update outputs to correspond to the new W */
            ocas.trn_err = 0;   /*  trn_err = sum_i [[y != y_i ]]                       */
            R = 0;
            for(i=0; i < nData; i++)
            {
                y2 = (uint32_t)data_y[i]-1;

                for(tmp=-LIBOCAS_PLUS_INF, y=0; y < nY; y++)
                {
                    output[LIBOCAS_INDEX(y,i,nY)] = old_output[LIBOCAS_INDEX(y,i,nY)]*(1-t1) + t1*output[LIBOCAS_INDEX(y,i,nY)];

                    if(y2 != y && tmp < output[LIBOCAS_INDEX(y,i,nY)])
                    {
                        ypred = y;
                        tmp = output[LIBOCAS_INDEX(y,i,nY)];
                    }
                }

                R += LIBOCAS_MAX(0,1+tmp - output[LIBOCAS_INDEX(y2,i,nY)]);
                if( tmp >= output[LIBOCAS_INDEX(y2,i,nY)])
                    ocas.trn_err ++;
            }

            ocas.Q_P = 0.5*sq_norm_W + C*R;


            /* get time and print status */
            ocas.ocas_time += get_time() - LS_start_time;

            //ocas_print(ocas);
            //fflush(stdout);

            Accuracy = evaluate_multiclass(test_X, test_y, nData_test);

            printf("ocas_iter:%d time:%lf pf:%lf df:%lf acc:%lf lstime:%lf pwl_time:%lf\n",
                   ocas.nIter, ocas.ocas_time, ocas.Q_P, ocas.Q_D,
                   Accuracy, LS_time, PWL_time);
            fprintf(fpresult,"%d\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n",
                    ocas.nIter,ocas.ocas_time, ocas.Q_P, ocas.Q_D,
                    Accuracy, LS_time, PWL_time);

            break;

            /* The ACUPA-M solver */
        case 2:
            if(DEBUG) // all the random index in set U saved in indexfile.
                fpinx = fopen("indexfile", "w");

            LS_start_time = get_time();
            memcpy( old_output, output, sizeof(double)*nData*nY );

            if( compute_output( output, user_data ) != 0)
            { // output=W'X
                ocas.exitflag=-2;
                goto cleanup;
            }

            //LS_start_time = get_time();
            A0 = sq_norm_W - 2*dot_prod_WoldW + sq_norm_oldW;
            B0 = dot_prod_WoldW - sq_norm_oldW;

            // compute all the Bis and Cis(i.e., A and B below) to
            // find the explicit PWL representation
            for(i=0; i < nData; i++)
            {
                y2 = (uint32_t)data_y[i]-1;

                for(y=0; y < nY; y++)
                {
                    A[LIBOCAS_INDEX(y,i,nY)] = C*(output[LIBOCAS_INDEX(y,i,nY)] - old_output[LIBOCAS_INDEX(y,i,nY)]
                            + old_output[LIBOCAS_INDEX(y2,i,nY)] - output[LIBOCAS_INDEX(y2,i,nY)]);
                    B[LIBOCAS_INDEX(y,i,nY)] = C*(old_output[LIBOCAS_INDEX(y,i,nY)] - old_output[LIBOCAS_INDEX(y2,i,nY)]
                            + (double)(y != y2));
                }
            }

            grad_sum = B0;
            numLambda = 0;

            PWL_start_time = get_time();
            for(i=0; i < nData; i++)
                findpwl(Add, Theta, &numLambda, &A[i*nY],
                        &B[i*nY], nY, &grad_sum);
            PWL_time = get_time() - PWL_start_time;

            /* Theta: all the active breakpoints(i.e., \lambda)
             * Add: sup g(\lam) - inf g(\lam)
             */

            GradVal = grad_sum;
            lam_k = 0;
            if(GradVal < 0)
            {
                uint32_t inx, k = 0, k_old, Ulength = numLambda;
                double lam_old;

                for(i=0; i<numLambda; i++)
                    U[i] = i;

                srand(time(NULL));
                for(s=0; s<numLambda; s++)
                {
                    inx = rand() % Ulength;// inx \in [0,...[U]-1]

                    k_old = k;
                    k = U[inx];  // Random choose k \in U.
                    lam_old = lam_k;
                    lam_k = Theta[k];

                    if(DEBUG)
                        fprintf(fpinx, "%d\t%d\t%lf\t%d\n",
                                s, numLambda, lam_k, inx);

                    int gi = 0, li = 0;
                    for(i=0; i<Ulength; i++){ // set G and L
                        double lambiUi = Theta[U[i]];
                        if(lambiUi >= lam_k){
                            G[gi] = U[i];
                            gi++;
                        }
                        else{
                            L[li] = U[i];
                            li++; 
                        }
                    }

                    if(lam_k >= lam_old){
                        GradVal += A0*(lam_k-lam_old);
                        for(i=0; i<li; i++)
                            GradVal += Add[L[i]];
                    }
                    else{
                        GradVal -= (Add[k_old] + A0*(lam_old-lam_k));
                        for(i=0; i<gi; i++)
                            GradVal -= Add[G[i]];
                    }

                    if(GradVal<0){
                        GradVal += Add[k]; // get sup g(lam_k)
                        if(GradVal>=0)
                            break;
                        else{
                            for(i=0, j=0; i<gi; i++){
                                if(G[i]==k)
                                    continue;
                                U[j] = G[i];
                                j++;
                            }
                            Ulength = gi-1;
                        }
                    }
                    else{ // GradVal>=0
                        if(li>0)
                            GradVal += Add[k];
                        for(i=0; i<li; i++)
                            U[i] = L[i];
                        Ulength = li;
                    }

                    if(Ulength == 0){
                        lam_k -= GradVal/A0;
                        break;
                    }
                }
                if(DEBUG)
                    fclose(fpinx);
            }
            /* end of the linesearch which outputs lam_k */
            LS_time += get_time() - LS_start_time;

            t = LIBOCAS_MAX(lam_k,0);
            t1 = t;                /* new (best so far) W */
            t2 = t+(1.0-t)*MU;   /* new cutting plane */


            //if(t == 0)
            //    t2 = MU;
            //else
            //    t2 = t1;


            /* update W to be the best so far solution */
            sq_norm_W = update_W( t1, user_data );

            /* the following code  computes a new cutting plane: */
            element_b = 0.0;    /*  element_b = R(old_W) - g'*old_W */
            /* new_cut[i] = argmax_i ( [[y != y_i]] + (w_y- w_y_i)'*x_i )  */
            for(i=0; i < nData; i++)
            {
                y2 = (uint32_t)data_y[i]-1;

                for(xi=-LIBOCAS_PLUS_INF, y=0; y < nY; y++)
                {
                    tmp = old_output[LIBOCAS_INDEX(y,i,nY)]*(1-t2) + t2*output[LIBOCAS_INDEX(y,i,nY)];
                    if(y2 != y && xi < tmp)
                    {
                        xi = tmp;
                        ypred = y;
                    }
                }

                tmp = old_output[LIBOCAS_INDEX(y2,i,nY)]*(1-t2) + t2*output[LIBOCAS_INDEX(y2,i,nY)];
                xi = LIBOCAS_MAX(0,xi+1-tmp);
                if(xi > 0)
                {
                    element_b++;
                    new_cut[i] = ypred;
                }
                else
                    new_cut[i] = y2;
            }

            /* compute Risk, class. error and update outputs to correspond to the new W */
            ocas.trn_err = 0;   /*  trn_err = sum_i [[y != y_i ]]                       */
            R = 0;
            for(i=0; i < nData; i++)
            {
                y2 = (uint32_t)data_y[i]-1;

                for(tmp=-LIBOCAS_PLUS_INF, y=0; y < nY; y++)
                {
                    output[LIBOCAS_INDEX(y,i,nY)] = old_output[LIBOCAS_INDEX(y,i,nY)]*(1-t1) + t1*output[LIBOCAS_INDEX(y,i,nY)];

                    if(y2 != y && tmp < output[LIBOCAS_INDEX(y,i,nY)])
                    {
                        ypred = y;
                        tmp = output[LIBOCAS_INDEX(y,i,nY)];
                    }
                }

                R += LIBOCAS_MAX(0,1+tmp - output[LIBOCAS_INDEX(y2,i,nY)]);
                if( tmp >= output[LIBOCAS_INDEX(y2,i,nY)])
                    ocas.trn_err ++;
            }

            ocas.Q_P = 0.5*sq_norm_W + C*R;

            /* get time and print status */
            ocas.ocas_time += get_time() - LS_start_time;

            //ocas_print(ocas);

            Accuracy = evaluate_multiclass(test_X, test_y, nData_test);

            printf("acupam_iter:%d time:%lf pf:%lf df:%lf acc:%lf lstime:%lf pwl_time:%lf\n",
                   ocas.nIter, ocas.ocas_time, ocas.Q_P, ocas.Q_D,
                   Accuracy, LS_time, PWL_time);
            fprintf(fpresult,"%d\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n",
                    ocas.nIter,ocas.ocas_time, ocas.Q_P, ocas.Q_D,
                    Accuracy, LS_time, PWL_time);
            fflush(stdout);

            break;

        }

        /* Stopping conditions */
        if( ocas.Q_P - ocas.Q_D <= TolRel*LIBOCAS_ABS(ocas.Q_P)) ocas.exitflag = 1;
        if( ocas.Q_P - ocas.Q_D <= TolAbs) ocas.exitflag = 2;
        if( ocas.Q_P <= QPBound) ocas.exitflag = 3;
        if( ocas.ocas_time >= MaxTime) ocas.exitflag = 4;
        if(ocas.nCutPlanes >= BufSize) ocas.exitflag = -1;

    } /* end of the main loop */

    fclose(fpresult);

cleanup:

    LIBOCAS_FREE(H);
    LIBOCAS_FREE(b);
    LIBOCAS_FREE(alpha);
    LIBOCAS_FREE(new_cut);
    LIBOCAS_FREE(I);
    LIBOCAS_FREE(diag_H);
    LIBOCAS_FREE(output);
    LIBOCAS_FREE(old_output);
    LIBOCAS_FREE(A);
    LIBOCAS_FREE(B);
    LIBOCAS_FREE(theta);
    LIBOCAS_FREE(Theta);
    LIBOCAS_FREE(sortedA);
    LIBOCAS_FREE(Add);

    ocas.ocas_time = get_time() - ocas_start_time;

    return(ocas);
}
