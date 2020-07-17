#include<iostream>
#include<cmath>
#include "matrix.h"
#include "lle.h"

using namespace std;

int main()
{
    double** X=new double*[20];
    for(int i=0;i<20;i++)
    {
        X[i]=new double[5];
        for(int j=0;j<5;j++)
        {
            if(i%2==0)
            X[i][j]=20*(rand()/(1.0+RAND_MAX))+20;
            else
            X[i][j]=20*(rand()/(1.0+RAND_MAX))+120;

        }
    }
    X=normalization(X,20,5);

    LLE* lle=new LLE(X,5,1,20);
    lle->update_weights(8000,0.001);
    lle->update_z(9600,0.0001,1e-6);

    double** Z=lle->get_Z();
    for(int i=0;i<20;i++)
    {
        for(int j=0;j<1;j++)
        cout<<Z[i][j]<<" ";
        cout<<endl;
    }
  
    return 0;
}
