#include "slsc_mex.h"
#include "mex.h"
mxArray *makelagmat(mwSize numrows, mwSize numcols, int maxlag) {
    
    // Declare variables
    mxArray *elems, *lagmat, *lag, *lagdist, *tmp;
    mxLogical *elem;
    int32 *lm, i, j, k;
    float *ld;
    
    // Set up matrices
    elems  = mxCreateLogicalMatrix(numrows,numcols);
    lagmat = mxCreateNumericMatrix(numrows*numcols,numrows*numcols,mxINT32_CLASS,mxREAL);
    elem   = mxGetLogicals(elems);
    lm     = (int32 *)mxGetData(lagmat);
    lag    = mxCreateCellMatrix(maxlag,1);
    
    for (i = 0; i < numrows*numcols; i++) {
        // Turn element i on
        elem[i] = true;
        // Find all elements' distances from element i
        mexCallMATLAB(1, &lagdist, 1, &elems, "bwdist");
        ld = (float *)mxGetData(lagdist);
        
        // Fill in a column of data, only upper triangle
        for (j = 0; j <= i; j++)
            lm[j + i*mxGetM(lagmat)] = (int32)(round(ld[j]) - 1); // zero indexed
        
        // Switch element i off again
        elem[i] = false;
    }
    mxArray *lagptr;
    int32 *dataptr;
    int count,count1;
    for (k = 0; k < maxlag; k++) {
        // Figure out how large cell k of lag should be
        count = 0;
        for (i = 0; i < numrows*numcols; i++) {
            for (j = 0; j <= i; j++) {
                if (lm[j + i*mxGetM(lagmat)] == k)
                    count++;
            }
        }
        // Fill in dataptr with locations of lm == k
        lagptr  = mxCreateNumericMatrix(count, 1, mxINT32_CLASS, mxREAL);
        dataptr = (int32 *)mxGetData(lagptr);
        count1 = 0;
        for (i = 0; i < numrows*numcols; i++) {
            for (j = 0; j <= i; j++) {
                if (lm[j + i*mxGetM(lagmat)] == k)
                    dataptr[count1++] = j + i*mxGetM(lagmat) + 1;
                if (count1 == count)
                    break;
            }
            if (count1 == count)
                break;
        }
        // Set this data to cell k of lag
        mxSetCell(lag, k, lagptr);
    }
    
    return lag;
}
