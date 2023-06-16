#include <vector>
#include "Lindsey.inl"
#include "math.h"
#include <iostream>
#include "Lindsey.h"

using namespace std;

void LindseyVals(double* xval, int nx, double* funval)
{
#pragma omp parallel for
    for (int i=0; i<nx; i++) 
    {
        //funval[i] = 0.;
        if (xval[i] > 10.5 || xval[i] < -10.5)
            continue;

        int xi = int(xval[i]*3);
        for (int j=-14+xi; j<=14+xi; j++) 
        {
            funval[i] += gauss_coeff[j+150] * exp(- pow( (xval[i] - j*1./3)/(2./3.), 2)/2.);
        }
    }
}

/*
int main(int argc, char * argv [])
{
    int nx = 300;
    double xval [nx];
    double funval [nx];

    //xval[0] = 0.;
    //LindseyVals(xval, nx, funval);
    //exit(0);
    for (int i=0; i<nx; i++) 
    {
        xval[i] = -12 + (24.)*i/299.;
    }

    LindseyVals(xval, nx, funval);

    cout.precision(18);   
    for (int i=0; i<nx; i++) 
        cout << i*1.<<"  "<<xval[i]<<"  "<<funval[i]<<endl;
}
*/