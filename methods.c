#include "methods.h"

// root finding via randomized median searching algorithm
int rootrmf(double A0,
            double B0,
            double *Bi,
            double *Ci,
            double *lambda,
            int nData)
{
    double *hpb, *hpc, *lambi;
    uint32_t *U, *G, *L; //index set for randomized median finding
    uint32_t num = 0; //record the num of -ci/bi>0;

    hpb = (double*) LIBOCAS_CALLOC(nData, sizeof(hpb[0]));
    hpc = (double*) LIBOCAS_CALLOC(nData, sizeof(hpc[0]));
    lambi = (double*) LIBOCAS_CALLOC(nData, sizeof(lambi[0]));
    U = (uint32_t*) LIBOCAS_CALLOC(nData, sizeof(uint32_t));
    G = (uint32_t*) LIBOCAS_CALLOC(nData, sizeof(uint32_t));
    L = (uint32_t*) LIBOCAS_CALLOC(nData, sizeof(uint32_t));


    double GradVal = 0;
    double GradVal0 = B0;
    double val;
    uint32_t s = 0;
    int i,j;
    for(i=0; i< nData; i++)
    {  // Choose Bi and Ci, and compute grad at 0.
        if(Bi[i] != 0)
            val = -Ci[i]/Bi[i];
        else
            val = -LIBOCAS_PLUS_INF;

        if (val>0){
            /*            hpi[num_Ui] = i;*/
            U[num] = num;  // RMF: U=[n]
            hpb[num] = Bi[i];
            hpc[num] = Ci[i];
            lambi[num] = val;

            num++;
        }
        if (Ci[i]>=0){
            if(Ci[i]==0 & Bi[i]<0)
                continue;
            GradVal0 += Bi[i];
        }
    }
    if(GradVal0>=0){
        //cout <<"gradval0>=0" <<endl;
        return s;
    }

    uint32_t inx, k=0, k_old, Ulength = num;
    double lam_old, lam_k;
    lam_k = *lambda;
    srand(time(NULL));
    for(s=0; s<num; s++)
    {

        inx = rand() % Ulength; // inx \in [0,...[U]-1]
        k_old = k;
        k = U[inx];  // Random choose k \in U.
        lam_old = lam_k;
        lam_k = lambi[k];
//        cout <<"s=" <<s <<", k=" <<k <<", lam_k=" <<lam_k;
//        cout <<", U=[";
//        for(i=0; i<Ulength; i++)
//            cout <<U[i] <<" ";
//        cout <<"]" <<endl;


        int gi=0, li=0;
        for(i=0; i<Ulength; i++){ // set G and L
            double lambiUi = lambi[U[i]];
            if(lambiUi >= lam_k) {
                G[gi] = U[i];
                gi++;  // gi is the number of G
            }
            else {
                L[li] = U[i];
                li++;  // li is the number of L
            }
        }

        if(s==0){
            GradVal = GradVal0 + A0*(lam_k-lam_old);
            for(i=0; i<li; i++)
                GradVal += fabs(hpb[L[i]]);
        }
        else{
            if(lam_k > lam_old){
                GradVal += fabs(hpb[k_old]) + A0*(lam_k-lam_old);
                for(i=0; i<li; i++)
                    GradVal += fabs(hpb[L[i]]);
            }
            else{
                GradVal -= A0*(lam_old-lam_k);
                for(i=0; i<gi; i++)
                    GradVal -= fabs(hpb[G[i]]);
            }
        }

        if(GradVal<0){
            if(GradVal + LIBOCAS_ABS(hpb[k]) >=0)
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

            if(Ulength==0){
                lam_k -= (GradVal + LIBOCAS_ABS(hpb[k]))/A0;
                break;
            }
        }
        else{ // GradVal>=0
            for(i=0; i<li; i++)
                U[i] = L[i];
            Ulength = li;

            if(Ulength==0){
                lam_k -= GradVal/A0;
                break;
            }
        }
    }

    *lambda = lam_k;

    LIBOCAS_FREE(hpb);
    LIBOCAS_FREE(hpc);
    LIBOCAS_FREE(lambi);
    LIBOCAS_FREE(U);
    LIBOCAS_FREE(G);
    LIBOCAS_FREE(L);

    return s+1;

}
