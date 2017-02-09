/*-----------------------------------------------------------------------
 * ocas_helper.c: Implementation of helper functions for the OCAS solver.
 *
 *-------------------------------------------------------------------- */

#define _FILE_OFFSET_BITS  64

#include <pthread.h>

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <sys/time.h>
#include <stdlib.h>
#include <time.h>

#include "lib_svmlight_format.h"
#include "libocas.h"
#include "ocas_helper.h"

mxArray *data_X, *test_X;
uint32_t nDim, nY, nDim_test;
int nData, nData_test;
double *data_y, *test_y;
uint16_t Method;
cutting_plane_buf_T sparse_A;
double *full_A;
double *W;
double *oldW;
double *new_a;

double *A0;
double W0;
double oldW0;
double X0;

/* parallelization via threads */
struct thread_params_output
{
	double* output;
	uint32_t start;
	uint32_t end;
};

struct thread_qsort
{
	double* output;
/*	uint32_t* index;*/
	double* data;
	uint32_t size;
};

struct thread_params_add
{
  double *new_a;
  uint32_t *new_cut;
  uint32_t start;
  uint32_t end;
};


typedef enum 
{ 
  FALSE = 0,
  TRUE = 1
} 
boolean;


//static int qsort_threads;
//static pthread_t* threads = NULL;
//static uint32_t* thread_slices = NULL;
//static int num_threads;
//static const int sort_limit=4096;
//static struct thread_params_output* params_output;
//static struct thread_params_add* params_add;

/* use multi-threads only if minimal number of examples to add is higher than the constant*/
//static const uint32_t MinimalParallelCutLenght = 100;


/*----------------------------------------------------------------------
  ----------------------------------------------------------------------*/
int full_add_nnw_constr(uint32_t idx, uint32_t nSel, void* user_data)
{
    full_A[LIBOCAS_INDEX(idx,nSel,nDim)] = 1.0;
    A0[nSel] = 0.0;

    return( 0 );
}


/*----------------------------------------------------------------------
  ----------------------------------------------------------------------*/
int sparse_add_nnw_constr(uint32_t idx, uint32_t nSel, void* user_data)
{
  sparse_A.nz_dims[nSel] = 1;
  sparse_A.index[nSel] = NULL;
  sparse_A.value[nSel] = NULL;
  sparse_A.index[nSel] = mxCalloc(1,sizeof(uint32_t));
  sparse_A.value[nSel] = mxCalloc(1,sizeof(double));
  if(sparse_A.index[nSel]==NULL || sparse_A.value[nSel]==NULL)
  {
      mxFree(sparse_A.index[nSel]);
      mxFree(sparse_A.value[nSel]);
      return(-1);
  }

  sparse_A.index[nSel][0] = idx;
  sparse_A.value[nSel][0] = 1.0;

  A0[nSel] = 0.0;
  return( 0 );
}


/*----------------------------------------------------------------------
  ----------------------------------------------------------------------*/
void clip_neg_W( uint32_t num_pw_constr, uint32_t *pw_idx, void* user_data )
{
    uint32_t i;
    for(i=0; i< num_pw_constr; i++ ) {
        W[LIBOCAS_INDEX(pw_idx[i],0,nDim)] = LIBOCAS_MAX(0,W[LIBOCAS_INDEX(pw_idx[i],0,nDim)]);
    }
}


/*----------------------------------------------------------------------
  sq_norm_W = sparse_compute_W( alpha, nSel ) does the following:

  oldW = W;
  W = sparse_A(:,1:nSel)'*alpha;
  sq_norm_W = W'*W;
  dp_WoldW = W'*oldW';

  ----------------------------------------------------------------------*/
void sparse_compute_W( double *sq_norm_W, 
                       double *dp_WoldW, 
                       double *alpha, 
                       uint32_t nSel, 
                       void* user_data )
{
  uint32_t i,j, nz_dims;

  memcpy(oldW, W, sizeof(double)*nDim ); 
  memset(W, 0, sizeof(double)*nDim);

  oldW0 = W0;
  W0 = 0;

  for(i=0; i < nSel; i++) {
    nz_dims = sparse_A.nz_dims[i];
    if(nz_dims > 0 && alpha[i] > 0) {
      for(j=0; j < nz_dims; j++) {
        W[sparse_A.index[i][j]] += alpha[i]*sparse_A.value[i][j];
      }
    }
    W0 += A0[i]*alpha[i];
  }

  *sq_norm_W = W0*W0;
  *dp_WoldW = W0*oldW0;
  for(j=0; j < nDim; j++) {
    *sq_norm_W += W[j]*W[j];
    *dp_WoldW += W[j]*oldW[j];
  }
  
  return;
}


/*----------------------------------------------------------------------
  sq_norm_W = sparse_compute_W( alpha, nSel ) does the following:

  oldW = W;
  W = sparse_A(:,1:nSel)'*alpha;
  sq_norm_W = W'*W;
  dp_WoldW = W'*oldW';

  ----------------------------------------------------------------------*/
void msvm_sparse_compute_W( double *sq_norm_W, 
                       double *dp_WoldW, 
                       double *alpha, 
                       uint32_t nSel, 
                       void* user_data )
{
  uint32_t i,j, nz_dims;

  memcpy(oldW, W, sizeof(double)*nY*nDim ); 
  memset(W, 0, sizeof(double)*nY*nDim );

  for(i=0; i < nSel; i++) {
    nz_dims = sparse_A.nz_dims[i];
    if(nz_dims > 0 && alpha[i] > 0) {
      for(j=0; j < nz_dims; j++) {
        W[sparse_A.index[i][j]] += alpha[i]*sparse_A.value[i][j];
      }
    }
  }

  *sq_norm_W = 0;
  *dp_WoldW = 0;
  for(j=0; j < nY*nDim; j++)
  {
    *sq_norm_W += W[j]*W[j];
    *dp_WoldW += W[j]*oldW[j];
  }
  
  return;
}


/*----------------------------------------------------------------------------------
  sq_norm_W = sparse_update_W( t ) does the following:

  W = oldW*(1-t) + t*W;
  sq_norm_W = W'*W;

  ---------------------------------------------------------------------------------*/
double msvm_update_W( double t, void* user_data )
{
  uint32_t j;
  double sq_norm_W;         

  sq_norm_W = 0;

  for(j=0; j < nY*nDim; j++) {
    W[j] = oldW[j]*(1-t) + t*W[j];
    sq_norm_W += W[j]*W[j];
  }          

  return( sq_norm_W );
}


/*-------------------------------------------------------------------------
  sq_norm_W = full_update_W( t ) does the following:

  W = oldW*(1-t) + t*W;
  sq_norm_W = W'*W;
---------------------------------------------------------------------------*/
double update_W( double t, void* user_data )
{
  uint32_t j;
  double sq_norm_W;         

  W0 = oldW0*(1-t) + t*W0;
  sq_norm_W = W0*W0;

  for(j=0; j <nDim; j++) {
    W[j] = oldW[j]*(1-t) + t*W[j];
    sq_norm_W += W[j]*W[j];
  }          

  return( sq_norm_W );
}


/*----------------------------------------------------------------------
  sq_norm_W = full_compute_W( alpha, nSel ) does the following:

  oldW = W;
  W = full_A(:,1:nSel)'*alpha;
  sq_norm_W = W'*W;
  dp_WoldW = W'*oldW';

  ----------------------------------------------------------------------*/
void msvm_full_compute_W( double *sq_norm_W, double *dp_WoldW, double *alpha, uint32_t nSel, void* user_data )
{
  uint32_t i,j;

  memcpy(oldW, W, sizeof(double)*nDim*nY ); 
  memset(W, 0, sizeof(double)*nDim*nY);

  for(i=0; i < nSel; i++) {
    if( alpha[i] > 0 ) {
      for(j=0; j< nDim*nY; j++ ) {
        W[j] += alpha[i]*full_A[LIBOCAS_INDEX(j,i,nDim*nY)];
      }

    }
  }

  *sq_norm_W = 0;
  *dp_WoldW = 0;
  for(j=0; j < nDim*nY; j++) {
    *sq_norm_W += W[j]*W[j];
    *dp_WoldW += W[j]*oldW[j];
  }

  return;
}


/*-----------------------------------------------------------------------
  Print statistics.
  -----------------------------------------------------------------------*/
void ocas_print(ocas_return_value_T value)
{
  mexPrintf("%4d: tim=%f, Q_P=%f, Q_D=%f, Q_P-Q_D=%f, 1-Q_D/Q_P=%f, nza=%4d, err=%.2f%%, qpf=%d\n",
            value.nIter,value.ocas_time, value.Q_P,value.Q_D,value.Q_P-value.Q_D,(value.Q_P-value.Q_D)/LIBOCAS_ABS(value.Q_P), 
            value.nNZAlpha, 100*(double)value.trn_err/(double)nData, value.qp_exitflag );
}

void ocas_print_null(ocas_return_value_T value)
{
  return;
}


/*-----------------------------------------------------------------------
  Get absolute time in seconds.
  -----------------------------------------------------------------------*/
double get_time()
{
	struct timeval tv;
	if (gettimeofday(&tv, NULL)==0)
		return tv.tv_sec+((double)(tv.tv_usec))/1e6;
	else
		return 0.0;
}


/*----------------------------------------------------------------------
  in-place computes sparse_mat(:,col)= alpha * sparse_mat(:,col)
  where alpha is a scalar and sparse_mat is Matlab sparse matrix.
  ----------------------------------------------------------------------*/
void mul_sparse_col(double alpha, mxArray *sparse_mat, uint32_t col)
{
	uint32_t nItems, ptr, i;
	INDEX_TYPE_T *Jc;
	double *Pr;

	Jc = mxGetJc(sparse_mat);
	Pr = mxGetPr(sparse_mat);

	nItems = Jc[col+1] - Jc[col];
	ptr = Jc[col];

	for(i=0; i < nItems; i++)
		Pr[ptr++]*=alpha;
}


/*----------------------------------------------------------------------
 It computes full_vec = full_vec + sparse_mat(:,col)
 where full_vec is a double array and sparse_mat is Matlab 
 sparse matrix.
  ----------------------------------------------------------------------*/
void add_sparse_col(double *full_vec, mxArray *sparse_mat, uint32_t col)
{
  uint32_t nItems, ptr, i, row;
  INDEX_TYPE_T *Ir, *Jc;
  double *Pr, val;
    
  Ir = mxGetIr(sparse_mat);
  Jc = mxGetJc(sparse_mat);
  Pr = mxGetPr(sparse_mat);

  nItems = Jc[col+1] - Jc[col];
  ptr = Jc[col];

  for(i=0; i < nItems; i++) {
    val = Pr[ptr];
    row = Ir[ptr++];

    full_vec[row] += val;
  }
}


/*----------------------------------------------------------------------
 It computes full_vec = full_vec - sparse_mat(:,col)
 where full_vec is a double array and sparse_mat is Matlab 
 sparse matrix.
  ----------------------------------------------------------------------*/
void subtract_sparse_col(double *full_vec, mxArray *sparse_mat, uint32_t col)
{
  uint32_t nItems, ptr, i, row;
  INDEX_TYPE_T *Ir, *Jc;
  double *Pr, val;
    
  Ir = mxGetIr(sparse_mat);
  Jc = mxGetJc(sparse_mat);
  Pr = mxGetPr(sparse_mat);

  nItems = Jc[col+1] - Jc[col];
  ptr = Jc[col];

  for(i=0; i < nItems; i++) {
    val = Pr[ptr];
    row = Ir[ptr++];

    full_vec[row] -= val;
  }
}


/*----------------------------------------------------------------------
 It computes dp = full_vec'*sparse_mat(:,col)
 where full_vec is a double array and sparse_mat is Matlab 
 sparse matrix.
  ----------------------------------------------------------------------*/
double dp_sparse_col(double *full_vec, mxArray *sparse_mat, uint32_t col)
{
  uint32_t nItems, ptr, i, row;
  INDEX_TYPE_T *Ir, *Jc;
  double *Pr, val, dp;

  Ir = mxGetIr(sparse_mat);
  Jc = mxGetJc(sparse_mat);
  Pr = mxGetPr(sparse_mat);

  dp = 0;
  nItems = Jc[col+1] - Jc[col];
  ptr = Jc[col];

  for(i=0; i < nItems; i++) {
    val = Pr[ptr];
    row = Ir[ptr++];

    dp += full_vec[row]*val;
  }

  return dp;
}


/*----------------------------------------------------------------------
  sq_norm_W = full_compute_W( alpha, nSel ) does the following:

  oldW = W;
  W = full_A(:,1:nSel)'*alpha;
  sq_norm_W = W'*W;
  dp_WoldW = W'*oldW';

  ----------------------------------------------------------------------*/
void full_compute_W( double *sq_norm_W, double *dp_WoldW, double *alpha, uint32_t nSel, void* user_data )
{
  uint32_t i,j;

  memcpy(oldW, W, sizeof(double)*nDim ); 
  memset(W, 0, sizeof(double)*nDim);

  oldW0 = W0;
  W0 = 0;

  for(i=0; i < nSel; i++) {
    if( alpha[i] > 0 ) {
      for(j=0; j< nDim; j++ ) {
        W[j] += alpha[i]*full_A[LIBOCAS_INDEX(j,i,nDim)];
      }

      W0 += A0[i]*alpha[i];
    }
  }

  *sq_norm_W = W0*W0;
  *dp_WoldW = W0*oldW0;
  for(j=0; j < nDim; j++) {
    *sq_norm_W += W[j]*W[j];
    *dp_WoldW += W[j]*oldW[j];
  }

  return;
}


/*=======================================================================
 OCAS helper functions for sorting numbers.
=======================================================================*/
static void swapf(double* a, double* b)
{
	double dummy=*b;
	*b=*a;
	*a=dummy;
}


/* sort arrays value and data according to value in ascending order */
int qsort_data(double* value, double* data, uint32_t size)
{
    if(size == 1)
      return 0;

	if (size==2)
	{
		if (value[0] > value[1])
		{
			swapf(&value[0], &value[1]);
/*			swapi(&data[0], &data[1]);*/
			swapf(&data[0], &data[1]);
		}
		return 0;
	}
	double split=value[size/2];

	uint32_t left=0;
	uint32_t right=size-1;

	while (left<=right)
	{
		while (value[left] < split)
			left++;
		while (value[right] > split)
			right--;

		if (left<=right)
		{
			swapf(&value[left], &value[right]);
/*			swapi(&data[left], &data[right]);*/
			swapf(&data[left], &data[right]);
			left++;
			right--;
		}
	}

	if (right+1> 1)
		qsort_data(value,data,right+1);

	if (size-left> 1)
		qsort_data(&value[left],&data[left], size-left);


    return 0;
}


/* ---------------------------------------------------------------------------------
This function loads regularization constants from a text file. Each line contains
a single constant. 
  ---------------------------------------------------------------------------------*/
int load_regconsts(char *fname, double **vec_C, uint32_t *len_vec_C, int verb)
{
  double C;
  char *line = NULL;
  int exitflag = 0;
  FILE *fid;

  if(verb) mexPrintf("Input file: %s\n", fname);

  fid = fopen(fname, "r");
  if(fid == NULL) {
    perror("fopen error ");
    mexPrintf("Cannot open input file.\n");
    exitflag = -1;
    goto clean_up;
  }

  line = mxCalloc(LIBSLF_MAXLINELEN, sizeof(char));
  if( line == NULL )
  {
    mexPrintf("Not enough memmory to allocate line buffer.\n");
    exitflag = -1;
    goto clean_up;
  }


  if(verb) mexPrintf("Counting regularization constants...");
  int go = 1;
  long line_cnt = 0;
  while(go) {
    
    if(fgets(line,LIBSLF_MAXLINELEN, fid) == NULL ) 
    {
      go = 0;
      if(verb)
      {
        if( (line_cnt % 1000) != 0) 
          mexPrintf(" %ld", line_cnt);
        mexPrintf(" EOF.\n");
      }

    }
    else
    {
      line_cnt ++;

      C = atof(line);
      
      if(verb)
      {
        if( (line_cnt % 1000) == 0) {
          mexPrintf(" %ld", line_cnt);
          fflush(NULL);
        }
      }
    }
  }
  
  *vec_C = (double*)mxCalloc(line_cnt, sizeof(double));
  if( vec_C == NULL )
  {
    mexPrintf("Not enough memmory to allocate vec_C.\n");
    exitflag = -1;
    goto clean_up;
  }

  fseek(fid, 0, SEEK_SET);

  if(verb) mexPrintf("Reading regularization constants...");
  go = 1;
  line_cnt = 0;
  while(go) 
  {
    
    if(fgets(line,LIBSLF_MAXLINELEN, fid) == NULL ) 
    {
      go = 0;
      if(verb)
      {
        if( (line_cnt % 1000) != 0) 
          mexPrintf(" %ld", line_cnt);
        mexPrintf(" EOF.\n");
      }

    }
    else
    {
      (*vec_C)[line_cnt] = atof(line);
      line_cnt ++;
      
      if(verb)
      {
        if( (line_cnt % 1000) == 0) {
          mexPrintf(" %ld", line_cnt);
          fflush(NULL);
        }
      }
    }
  }
  
  fclose(fid);
  *len_vec_C = line_cnt;

clean_up:
  mxFree(line);

  return(exitflag); 
}



/* --------------------------------------------------------------------------------- 
This function loads SVMlight data file to sparse matlab matrix data_X and 
dense vector data_y which both are assumed to global variables.
  ---------------------------------------------------------------------------------*/
int load_svmlight_file(char *fname, int verb)
{
  char *line = NULL;
  FILE *fid;
  double *feat_val = NULL;
  double sparse_memory_requirement, full_memory_requirement;
  uint32_t *feat_idx = NULL;
  long nnzf;
  int max_dim = 0;
  long j;
  uint64_t nnz = 0;
  mwSize *irs = NULL, *jcs = NULL;
  int exitflag = 0;
  double *sr = NULL;
  
/*  mexPrintf("Input file: %s\n", fname);*/

  fid = fopen(fname, "r");
  if(fid == NULL) {
    perror("fopen error ");
    mexPrintf("Cannot open input file.\n");
    exitflag = -1;
    goto clean_up;
  }

  line = mxCalloc(LIBSLF_MAXLINELEN, sizeof(char));
  if( line == NULL )
  {
    mexPrintf("Not enough memmory to allocate line buffer.\n");
    exitflag = -1;
    goto clean_up;
  }

  feat_idx = mxCalloc(LIBSLF_MAXLINELEN, sizeof(uint32_t));
  if( feat_idx == NULL )
  {
    mexPrintf("Not enough memmory to allocate feat_idx.\n");
    exitflag = -1;
    goto clean_up;
  }

  feat_val = mxCalloc(LIBSLF_MAXLINELEN, sizeof(double));
  if( feat_val == NULL )
  {
    mexPrintf("Not enough memmory to allocate feat_val.\n");
    exitflag = -1;
    goto clean_up;
  }

  if(verb) mexPrintf("Analysing input data...");
  int label;
  int go = 1;
  long line_cnt = 0;

  while(go) {
    
    if(fgets(line,LIBSLF_MAXLINELEN, fid) == NULL ) 
    {
      go = 0;
      if(verb)
      {
        if( (line_cnt % 1000) != 0) 
          mexPrintf(" %ld", line_cnt);
        mexPrintf(" EOF.\n");
      }

    }
    else
    {
      line_cnt ++;
      nnzf = svmlight_format_parse_line(line, &label, feat_idx, feat_val);
      
      if(nnzf == -1) 
      {
         mexPrintf("Parsing error on line %ld .\n", line_cnt);
         mexPrintf("Probably defective input file.\n");
         exitflag = -1;
         goto clean_up;
      }

      max_dim = LIBOCAS_MAX(max_dim,feat_idx[nnzf-1]);
      nnz += nnzf;
      
      if(verb)
      {
        if( (line_cnt % 1000) == 0) {
          mexPrintf(" %ld", line_cnt);
          fflush(NULL);
        }
      }
    }
  }

  fclose(fid);  
  if(verb)
  {
    mexPrintf("Data statistics:\n");
    mexPrintf("# of examples: %ld\n", line_cnt);
    mexPrintf("dimensionality: %d\n", max_dim);
    mexPrintf("nnz: %ld, density: %f%%\n", (long)nnz, 100*(double)nnz/((double)max_dim*(double)line_cnt));
  }

  sparse_memory_requirement = ((double)nnz*((double)sizeof(double)+(double)sizeof(mwSize)))/(1024.0*1024.0);
  full_memory_requirement = sizeof(double)*(double)max_dim*(double)line_cnt/(1024.0*1024.0);

  if(verb)
  {
    mexPrintf("Memory requirements for sparse matrix: %.3f MB\n", sparse_memory_requirement);
    mexPrintf("Memory requirements for full matrix: %.3f MB\n", full_memory_requirement);
  }

  if( full_memory_requirement < sparse_memory_requirement)
  {
    if(verb)
      mexPrintf("Full matrix represenation used.\n");

    data_X = mxCreateDoubleMatrix(max_dim, line_cnt, mxREAL);

    if( data_X == NULL)
    {
      mexPrintf("Not enough memory to allocate data_X .\n");
      exitflag = -1;
      goto clean_up;
    }

  }
  else
  {
    if(verb)
      mexPrintf("Sparse matrix represenation used.\n");

    data_X = mxCreateSparse(max_dim, line_cnt, nnz, mxREAL);
    if( data_X == NULL)
    {
      mexPrintf("Not enough memory to allocate data_X .\n");
      exitflag = -1;
      goto clean_up;
    }

    sr  = mxGetPr(data_X);
    irs = (mwSize*)mxGetIr(data_X);
    jcs = (mwSize*)mxGetJc(data_X);

  }


/*  mexPrintf("Required memory: %.3f MB\n", */
/*    ((double)nnz*((double)sizeof(double)+(double)sizeof(mwSize)))/(1024.0*1024.0));*/

  /*---------------------------------------------*/

  data_y = mxCalloc(line_cnt, sizeof(double));
  if(data_y == NULL)
  {
    mexPrintf("Not enough memory to allocate data_y.\n");
    exitflag = -1;
    goto clean_up;
  }

  fid = fopen(fname, "r");
  if(fid == NULL) {
    perror("fopen error ");
    mexPrintf("Cannot open input file.\n");
    exitflag = -1;
    goto clean_up;
  }

  if(verb)
    mexPrintf("Reading examples...");
  
  go = 1;
  line_cnt = 0;
  long k=0;
  while(go) {
    if(fgets(line,LIBSLF_MAXLINELEN, fid) == NULL ) 
    {
      go = 0;
      if(verb)
      {
        if( (line_cnt % 1000) != 0) 
          mexPrintf(" %ld", line_cnt);
        mexPrintf(" EOF.\n");
      }
    }
    else
    {
      line_cnt ++;
      nnzf = svmlight_format_parse_line(line, &label, feat_idx, feat_val);
      
      if(nnzf == -1) 
      {
         mexPrintf("Parsing error on line %ld .\n", line_cnt);
         mexPrintf("Defective input file.\n");
         exitflag = -1;
         goto clean_up;
      }

      data_y[line_cnt-1] = (double)label;

      if( mxIsSparse( data_X) )
      {
        jcs[line_cnt-1] = k;

        for(j = 0; j < nnzf; j++) {
          sr[k] = feat_val[j];
          irs[k] = feat_idx[j]-1;
          k++;
        }
      }
      else
      {
        double *ptr = mxGetPr(data_X);
        for(j=0; j < nnzf; j++ ) {
          ptr[LIBOCAS_INDEX(feat_idx[j]-1,line_cnt-1,max_dim)] = feat_val[j];
        }

      }
      
      if(verb)
      {
        if( (line_cnt % 1000) == 0) {
          mexPrintf(" %ld", line_cnt);
          fflush(NULL);
        }
      }
    }
  }

  fclose(fid);  

  if( mxIsSparse( data_X) )
    jcs[line_cnt] = k;

/*  mexPrintf("\n");*/

  if(verb)
    mexPrintf("Leaving svmlight reading function.\n");

  exitflag = 0;

clean_up:

  mxFree(line);
  mxFree(feat_val);
  mxFree(feat_idx);

  return(exitflag);
}

/*
 * This function loads SVMlight data file to test_X and test_y which both
 * are assumed global variables.
 */
int load_svmlight_file_test(char *fname, int verb)
{
  char *line = NULL;
  FILE *fid;
  double *feat_val = NULL;
  double sparse_memory_requirement, full_memory_requirement;
  uint32_t *feat_idx = NULL;
  long nnzf;
  int max_dim = 0;
  long j;
  uint64_t nnz = 0;
  mwSize *irs = NULL, *jcs = NULL;
  int exitflag = 0;
  double *sr = NULL;

/*  mexPrintf("Input file: %s\n", fname);*/

  fid = fopen(fname, "r");
  if(fid == NULL) {
    perror("fopen error ");
    mexPrintf("Cannot open input testing file.\n");
    exitflag = -1;
    goto clean_up;
  }

  line = mxCalloc(LIBSLF_MAXLINELEN, sizeof(char));
  if( line == NULL )
  {
    mexPrintf("Not enough memmory to allocate line buffer.\n");
    exitflag = -1;
    goto clean_up;
  }

  feat_idx = mxCalloc(LIBSLF_MAXLINELEN, sizeof(uint32_t));
  if( feat_idx == NULL )
  {
    mexPrintf("Not enough memmory to allocate feat_idx.\n");
    exitflag = -1;
    goto clean_up;
  }

  feat_val = mxCalloc(LIBSLF_MAXLINELEN, sizeof(double));
  if( feat_val == NULL )
  {
    mexPrintf("Not enough memmory to allocate feat_val.\n");
    exitflag = -1;
    goto clean_up;
  }

  if(verb) mexPrintf("Analysing input testing data...");
  int label;
  int go = 1;
  long line_cnt = 0;

  while(go) {

    if(fgets(line,LIBSLF_MAXLINELEN, fid) == NULL )
    {
      go = 0;
      if(verb)
      {
        if( (line_cnt % 1000) != 0)
          mexPrintf(" %ld", line_cnt);
        mexPrintf(" EOF.\n");
      }

    }
    else
    {
      line_cnt ++;
      nnzf = svmlight_format_parse_line(line, &label, feat_idx, feat_val);

      if(verb)
      {
        if( (line_cnt % 1000) == 0) {
          mexPrintf(" %ld", line_cnt);
          fflush(NULL);
        }
      }

      if(nnzf == -1)
      {
         mexPrintf("Warning: Probably defective input file on line %ld .\n", line_cnt);
         //mexPrintf("Probably defective input file.\n");
         //exitflag = -1;
         //goto clean_up;
         continue;
      }

      max_dim = LIBOCAS_MAX(max_dim,feat_idx[nnzf-1]);
      nnz += nnzf;
    }
  }

  fclose(fid);
  if(verb)
  {
    mexPrintf("Testing Data statistics:\n");
    mexPrintf("# of examples: %ld\n", line_cnt);
    mexPrintf("dimensionality: %d\n", max_dim);
    mexPrintf("nnz: %ld, density: %f%%\n", (long)nnz, 100*(double)nnz/((double)max_dim*(double)line_cnt));
  }

  sparse_memory_requirement = ((double)nnz*((double)sizeof(double)+(double)sizeof(mwSize)))/(1024.0*1024.0);
  full_memory_requirement = sizeof(double)*(double)max_dim*(double)line_cnt/(1024.0*1024.0);

  if(verb)
  {
    mexPrintf("Memory requirements for sparse matrix: %.3f MB\n", sparse_memory_requirement);
    mexPrintf("Memory requirements for full matrix: %.3f MB\n", full_memory_requirement);
  }

  if( full_memory_requirement < sparse_memory_requirement)
  {
    if(verb)
      mexPrintf("Full matrix represenation used.\n");

    test_X = mxCreateDoubleMatrix(max_dim, line_cnt, mxREAL);

    if( test_X == NULL)
    {
      mexPrintf("Not enough memory to allocate test_X .\n");
      exitflag = -1;
      goto clean_up;
    }

  }
  else
  {
    if(verb)
      mexPrintf("Sparse matrix represenation used.\n");

    test_X = mxCreateSparse(max_dim, line_cnt, nnz, mxREAL);
    if( test_X == NULL)
    {
      mexPrintf("Not enough memory to allocate test_X .\n");
      exitflag = -1;
      goto clean_up;
    }

    sr  = mxGetPr(test_X);
    irs = (mwSize*)mxGetIr(test_X);
    jcs = (mwSize*)mxGetJc(test_X);

  }


/*  mexPrintf("Required memory: %.3f MB\n", */
/*    ((double)nnz*((double)sizeof(double)+
 *    (double)sizeof(mwSize)))/(1024.0*1024.0));*/

  /*---------------------------------------------*/

  test_y = mxCalloc(line_cnt, sizeof(double));
  if(test_y == NULL)
  {
    mexPrintf("Not enough memory to allocate test_y.\n");
    exitflag = -1;
    goto clean_up;
  }

  fid = fopen(fname, "r");
  if(fid == NULL) {
    perror("fopen error ");
    mexPrintf("Cannot open input testing file.\n");
    exitflag = -1;
    goto clean_up;
  }

  if(verb)
    mexPrintf("Reading examples...");

  go = 1;
  line_cnt = 0;
  long k=0;
  while(go) {
    if(fgets(line,LIBSLF_MAXLINELEN, fid) == NULL )
    {
      go = 0;
      if(verb)
      {
        if( (line_cnt % 1000) != 0)
          mexPrintf(" %ld", line_cnt);
        mexPrintf(" EOF.\n");
      }
    }
    else
    {
      line_cnt ++;
      nnzf = svmlight_format_parse_line(line, &label, feat_idx, feat_val);

      if(nnzf == -1)
      {
         mexPrintf("Warning: Probably defective input file on line %ld .\n", line_cnt);
         //mexPrintf("Probably defective input file.\n");
         //exitflag = -1;
         //goto clean_up;
         //continue;
      }

      test_y[line_cnt-1] = (double)label;

      if( mxIsSparse( test_X) )
      {
        jcs[line_cnt-1] = k;

        for(j = 0; j < nnzf; j++) {
          sr[k] = feat_val[j];
          irs[k] = feat_idx[j]-1;
          k++;
        }
      }
      else
      {
        double *ptr = mxGetPr(test_X);
        for(j=0; j < nnzf; j++ ) {
          ptr[LIBOCAS_INDEX(feat_idx[j]-1,line_cnt-1,max_dim)] = feat_val[j];
        }

      }

      if(verb)
      {
        if( (line_cnt % 1000) == 0) {
          mexPrintf(" %ld", line_cnt);
          fflush(NULL);
        }
      }
    }
  }

  fclose(fid);

  if( mxIsSparse( test_X) )
    jcs[line_cnt] = k;

/*  mexPrintf("\n");*/

  if(verb)
    mexPrintf("Leaving svmlight reading function.\n");

  exitflag = 0;

clean_up:

  mxFree(line);
  mxFree(feat_val);
  mxFree(feat_idx);

  return(exitflag);
}



/*----------------------------------------------------------------------
 Compute area under ROC (1st class label[i]==1; 2nd class label[i] != 1).
  ----------------------------------------------------------------------*/
double compute_auc(double *score, int *label, uint32_t nData)
{
  double *sorted_score = NULL;
  double *sorted_lab = NULL;
  uint32_t i;
  uint32_t neg, pos;
  double auc = -1;

  sorted_score = mxCalloc(nData, sizeof(double));
  if( sorted_score == NULL ) {
      mexPrintf("Not enough memmory to allocate sorted_score when computing AUC.");
      goto clean_up;
  }

  sorted_lab = mxCalloc(nData, sizeof(double));
  if( sorted_lab == NULL )
  {
      mexPrintf("Not enough memmory to allocate sorted_lab when computing AUC.");
      goto clean_up;
  }

  for(i=0; i < nData; i++)
    if(label[i] == 1) sorted_lab[i] = 1.0; else sorted_lab[i] = 0.0;


  memcpy(sorted_score,score,sizeof(double)*nData);

  qsort_data(sorted_score, sorted_lab, nData);

  pos = 0;
  neg = 0;
  auc = 0;

  for(i = 0; i < nData; i++)
  {
    if(sorted_lab[i] ==1.0 )
    {
      pos ++;
    }
    else
    {
      neg ++;
      auc += (double)pos;
    }
  }
  auc = 1 - auc/((double)neg*(double)pos);

clean_up:
  mxFree(sorted_score);
  mxFree(sorted_lab);

  return(auc);
}

