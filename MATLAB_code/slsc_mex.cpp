/*
 * ========================================================================
 * Name        : slsc_mex.cpp
 * Author      : Dongwoon Hyun
 * ========================================================================
 *
 * Usage: cc = slsc_mex(datatype,signal,lag,kernelsz,downsamp);
 *
 * Performs the following calculation:
 * r = resid*resid', which can be split into real and imag
 * r_i = resid_i*resid_i' - resid_q*resid_q'  (REAL)
 * r_q = resid_q*resid_i' + resid_i*resid_q'  (IMAG)
 * where resid is a kernel of data that is zero-meaned in the sample direction.
 *
 * ssyr(k)_ is used whenever possible to take advantage of symmetry.
 * sgemm_ is used otherwise.
 */
#include "slsc_mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    int size_rhs;
    if ( (!mxIsCell(prhs[1]) && nrhs > 4 ) || (mxIsCell(prhs[1]) && nrhs > 3) )
        size_rhs = 4;
    else if ( (!mxIsCell(prhs[1]) && nrhs == 4) || (mxIsCell(prhs[1]) && nrhs == 3) )
        size_rhs = 3;
    else
        mexErrMsgTxt("Incorrect number of input arguments. See help slsc_mex\n");
    const mxArray *rhs[5]; // One more than the maximum actually needed, just to be safe
    rhs[0] = prhs[0];
    
    // Argument check, determination of scan convert mode //
    if (!mxIsCell(prhs[1])) {
        if (nrhs < 4 || nrhs > 6)
            mexErrMsgTxt("Incorrect number of input arguments. See help slsc_mex\n");
        // Make lagmat
        int numrows, numcols, maxlag;
        numrows = (int)mxGetScalar(prhs[1]);
        numcols = (int)mxGetScalar(prhs[2]);
        if (numrows * numcols != (int)(mxGetDimensions(prhs[0]))[1])
            mexErrMsgTxt("Dimension 2 of input data (#elements) does not equal numrows*numcols.");
        maxlag  = (int)round(sqrt((float)(numrows*numrows+numcols*numcols)))-1;
        if (nrhs == 6) {
            maxlag = (int)mxGetScalar(prhs[5]) < maxlag ? (int)mxGetScalar(prhs[5]) : maxlag;
            if (maxlag < 0)
                maxlag = 0;
        }
        rhs[1] = makelagmat(numrows, numcols, maxlag);
        rhs[2] = prhs[3];
        if (nrhs == 5 || nrhs == 6)
            rhs[3] = prhs[4];
    }
    else {
        if (nrhs >= 3) {
            rhs[1] = prhs[1];
            rhs[2] = prhs[2];
        }
        if (nrhs == 4)
            rhs[3] = prhs[3];
        if (nrhs < 3 || nrhs > 4)
            mexErrMsgTxt("Incorrect number of input arguments. See help slsc_mex\n");
    }
    
    // Call function
    if (!mxIsComplex(prhs[0])) {
//         mexPrintf("RF data detected...\n");
//         mexEvalString("drawnow");
        if (mxGetClassID(prhs[0]) == mxSINGLE_CLASS)
            RFdataSINGLE<float>(nlhs, plhs, size_rhs, rhs);
        else if (mxGetClassID(prhs[0]) == mxDOUBLE_CLASS)
            RFdataDOUBLE<double>(nlhs, plhs, size_rhs, rhs);
        else if (mxGetClassID(prhs[0]) == mxINT8_CLASS)
            RFdataSINGLE<int8  >(nlhs, plhs, size_rhs, rhs);
        else if (mxGetClassID(prhs[0]) == mxINT16_CLASS)
            RFdataSINGLE<int16 >(nlhs, plhs, size_rhs, rhs);
        else if (mxGetClassID(prhs[0]) == mxINT32_CLASS)
            RFdataSINGLE<int32 >(nlhs, plhs, size_rhs, rhs);
        else if (mxGetClassID(prhs[0]) == mxINT64_CLASS)
            RFdataDOUBLE<int64 >(nlhs, plhs, size_rhs, rhs);
        else if (mxGetClassID(prhs[0]) == mxUINT8_CLASS)
            RFdataSINGLE<uint8 >(nlhs, plhs, size_rhs, rhs);
        else if (mxGetClassID(prhs[0]) == mxUINT16_CLASS)
            RFdataSINGLE<uint16>(nlhs, plhs, size_rhs, rhs);
        else if (mxGetClassID(prhs[0]) == mxUINT32_CLASS)
            RFdataSINGLE<uint32>(nlhs, plhs, size_rhs, rhs);
        else if (mxGetClassID(prhs[0]) == mxUINT64_CLASS)
            RFdataDOUBLE<uint64>(nlhs, plhs, size_rhs, rhs);
        else
            mexErrMsgTxt("Unsupported input data type");
    }
    else {
//         mexPrintf("IQ data detected...\n");
//         mexEvalString("drawnow");
        if (mxGetClassID(prhs[0]) == mxSINGLE_CLASS)
            IQdataSINGLE<float >(nlhs, plhs, size_rhs, rhs);
        else if (mxGetClassID(prhs[0]) == mxDOUBLE_CLASS)
            IQdataDOUBLE<double>(nlhs, plhs, size_rhs, rhs);
        else if (mxGetClassID(prhs[0]) == mxINT8_CLASS)
            IQdataSINGLE<int8  >(nlhs, plhs, size_rhs, rhs);
        else if (mxGetClassID(prhs[0]) == mxINT16_CLASS)
            IQdataSINGLE<int16 >(nlhs, plhs, size_rhs, rhs);
        else if (mxGetClassID(prhs[0]) == mxINT32_CLASS)
            IQdataSINGLE<int32 >(nlhs, plhs, size_rhs, rhs);
        else if (mxGetClassID(prhs[0]) == mxINT64_CLASS)
            IQdataDOUBLE<int64 >(nlhs, plhs, size_rhs, rhs);
        else if (mxGetClassID(prhs[0]) == mxUINT8_CLASS)
            IQdataSINGLE<uint8 >(nlhs, plhs, size_rhs, rhs);
        else if (mxGetClassID(prhs[0]) == mxUINT16_CLASS)
            IQdataSINGLE<uint16>(nlhs, plhs, size_rhs, rhs);
        else if (mxGetClassID(prhs[0]) == mxUINT32_CLASS)
            IQdataSINGLE<uint32>(nlhs, plhs, size_rhs, rhs);
        else if (mxGetClassID(prhs[0]) == mxUINT64_CLASS)
            IQdataDOUBLE<uint64>(nlhs, plhs, size_rhs, rhs);
        else
            mexErrMsgTxt("Unsupported input data type");    }
    
    if (!mxIsCell(prhs[1]))
        mxDestroyArray((mxArray *)rhs[1]);
}

//int round(float a)  { return a > 0 ? (int)floor(a+0.5) : (int)ceil(a-0.5); }
