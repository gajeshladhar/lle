double** mat_mul(double** mat1,double** mat2,int r1,int c1,int r2,int c2)
{

    double** result;
    result=new double*[r1];
    for(int i=0;i<r1;i++)
    result[i]=new double[c2];
    double sum=0;

    for (int row = 0; row < r1; row++)
    {
        for (int col = 0; col < c2; col++)
        {
            sum = 0;

           
            for (int i = 0; i < c1; i++)
            {
                sum += (*(*(mat1 + row) + i)) * (*(*(mat2 + i) + col));
            }


            *(*(result + row) + col) = sum;
        }
    }

 return result;

}

double** mat_add(double** mat1,double** mat2,int r1,int c1,int r2,int c2)
{
    double** result;
    result=new double*[r1];
    for(int i=0;i<r1;i++)
    result[i]=new double[c1];

    if(r1==r2 && c1==c2)
    {
        for(int i=0;i<r1;i++)
        {
            for(int j=0;j<c1;j++)
            result[i][j]=mat1[i][j]+mat2[i][j];
        }
    }
    else if(r1==r2 && c2==1)
    {
        for(int i=0;i<r1;i++)
            {
                for(int j=0;j<c1;j++)
                result[i][j]=mat1[i][j]+mat2[i][0];
            }

    }

    else if(c1==r2 && c2==1)
    {
        for(int i=0;i<r1;i++)
            {
                for(int j=0;j<c1;j++)
                result[i][j]=mat1[i][j]+mat2[j][0];
            }

    }


    return result;

}

double** mat_add(double** mat1,double mat2,int r1,int c1)
{
    double** result;
    result=new double*[r1];
    for(int i=0;i<r1;i++)
    result[i]=new double[c1];
    
    for(int i=0;i<r1;i++)
    {
        for(int j=0;j<c1;j++)
         result[i][j]=mat1[i][j]+mat2;
    }

    return result;
}
double** mat_add(double mat1,double** mat2,int r2,int c2)
{
    double** result;
    result=new double*[r2];
    for(int i=0;i<r2;i++)
    result[i]=new double[c2];
    
    for(int i=0;i<r2;i++)
    {
        for(int j=0;j<c2;j++)
         result[i][j]=mat2[i][j]+mat1;
    }

    return result;
}



double** mat_sub(double** mat1,double mat2,int r1,int c1)
{
    double** result;
    result=new double*[r1];
    for(int i=0;i<r1;i++)
    result[i]=new double[c1];
    
    for(int i=0;i<r1;i++)
    {
        for(int j=0;j<c1;j++)
         result[i][j]=mat1[i][j]-mat2;
    }

    return result;
}
double** mat_sub(double mat1,double** mat2,int r2,int c2)
{
    double** result;
    result=new double*[r2];
    for(int i=0;i<r2;i++)
    result[i]=new double[c2];
    
    for(int i=0;i<r2;i++)
    {
        for(int j=0;j<c2;j++)
         result[i][j]=mat1-mat2[i][j];
    }

    return result;
}


double** mat_ele_mul(double** mat1,double mat2,int r1,int c1)
{
    double** result;
    result=new double*[r1];
    for(int i=0;i<r1;i++)
    result[i]=new double[c1];
    
    for(int i=0;i<r1;i++)
    {
        for(int j=0;j<c1;j++)
         result[i][j]=mat1[i][j]*mat2;
    }

    return result;
}
double** mat_ele_mul(double mat1,double** mat2,int r2,int c2)
{
    double** result;
    result=new double*[r2];
    for(int i=0;i<r2;i++)
    result[i]=new double[c2];
    
    for(int i=0;i<r2;i++)
    {
        for(int j=0;j<c2;j++)
         result[i][j]=mat2[i][j]*mat1;
    }

    return result;
}


double** mat_divide(double** mat1,double mat2,int r1,int c1)
{
    double** result;
    result=new double*[r1];
    for(int i=0;i<r1;i++)
    result[i]=new double[c1];
    
    for(int i=0;i<r1;i++)
    {
        for(int j=0;j<c1;j++)
         result[i][j]=mat1[i][j]/(1.0*mat2+1e-5);
    }

    return result;
}
double** mat_divide(double mat1,double** mat2,int r2,int c2)
{
    double** result;
    result=new double*[r2];
    for(int i=0;i<r2;i++)
    result[i]=new double[c2];
    
    for(int i=0;i<r2;i++)
    {
        for(int j=0;j<c2;j++)
         result[i][j]=(1.0*mat1)/mat2[i][j];
    }

    return result;
}



double** mat_sub(double** mat1,double** mat2,int r1,int c1,int r2,int c2)
{
    double** result;
    result=new double*[r1];
    for(int i=0;i<r1;i++)
    result[i]=new double[c1];

    if(r1==r2 && c1==c2)
    {
        for(int i=0;i<r1;i++)
        {
            for(int j=0;j<c1;j++)
            result[i][j]=mat1[i][j]-mat2[i][j];
        }
    }
    else if(r1==r2 && c2==1)
    {
        for(int i=0;i<r1;i++)
            {
                for(int j=0;j<c1;j++)
                result[i][j]=mat1[i][j]-mat2[i][0];
            }

    }

    return result;

}

double** mat_ele_mul(double** mat1,double** mat2,int r1,int c1,int r2,int c2)
{
    double** result;
    result=new double*[r1];
    for(int i=0;i<r1;i++)
    result[i]=new double[c1];

    if(r1==r2 && c1==c2)
    {
        for(int i=0;i<r1;i++)
        {
            for(int j=0;j<c1;j++)
            result[i][j]=mat1[i][j]*mat2[i][j];
        }
    }
    else if(r1==r2 && c2==1)
    {
        for(int i=0;i<r1;i++)
            {
                for(int j=0;j<c1;j++)
                result[i][j]=mat1[i][j]*mat2[i][0];
            }

    }
    else if(c1==c2 && r2==1)
    {
        for(int i=0;i<r1;i++)
            {
                for(int j=0;j<c1;j++)
                result[i][j]=mat1[i][j]*mat2[0][j];
            }

    }

    return result;

}

double** mat_divide(double** mat1,double** mat2,int r1,int c1,int r2,int c2)
{
    double** result;
    result=new double*[r1];
    for(int i=0;i<r1;i++)
    result[i]=new double[c1];

    if(r1==r2 && c1==c2)
    {
        for(int i=0;i<r1;i++)
        {
            for(int j=0;j<c1;j++)
            result[i][j]=mat1[i][j]/(1.0*mat2[i][j]);
        }
    }
    else if(r1==r2 && c2==1)
    {
        for(int i=0;i<r1;i++)
            {
                for(int j=0;j<c1;j++)
                result[i][j]=mat1[i][j]/(1.0*mat2[i][0]);
            }

    }
    else if(c1==c2 && r2==1)
    {
        for(int i=0;i<r1;i++)
            {
                for(int j=0;j<c1;j++)
                result[i][j]=mat1[i][j]/(1.0*mat2[0][j]);
            }

    }

    return result;

}

double** mat_transpose(double** mat,int r,int c)
{
    double** result=new double*[c];
    int i, j; 
    for(i=0;i<c;i++)
    result[i]=new double[r];

    
    for (i = 0; i < c; i++) 
        for (j = 0; j < r; j++) 
            result[i][j] = mat[j][i]; 

    return result;
}

double** mat_sum(double** mat,int r,int c,int axis)
{
    double** result;
    double sum=0;
    if(axis==0)
    {
    result=new double*[1];
    result[0]=new double[c];
    

    for(int j=0;j<c;j++)
    {
        for(int i=0;i<r;i++)
        {
            sum+=mat[i][j];
        }
        result[0][j]=sum;
    }
    }
    else if(axis==1)
    {
        result=new double*[r];
        for(int i=0;i<r;i++)
        result[i]=new double[1];

        for(int i=0;i<r;i++)
        {
            sum=0;
            for(int j=0;j<c;j++)
            sum+=mat[i][j];
            result[i][0]=sum;

        }
    }

    return result;
}

double** normalization(double** X,int r,int c)
{
  double* mean=new double[c];
  double* std=new double[c];
  
  for(int i=0;i<c;i++)
  {
      mean[i]=0;
      std[i]=0;
  }

  for(int i=0;i<r;i++)
  {
      for(int j=0;j<c;j++)
      mean[j]+=(X[i][j]/(1.0*r));
      
  }

  for(int i=0;i<r;i++)
  {
      for(int j=0;j<c;j++)
      std[j]+=((X[i][j]-mean[j])*(X[i][j]-mean[j])/(1.0*r) );
      
      
  }

  for(int i=0;i<c;i++)
  std[i]=sqrt(std[i]);

  for(int i=0;i<r;i++)
  {
      for(int j=0;j<c;j++)
      X[i][j]=(X[i][j]-mean[j])/(1.0*std[j]+1e-5);
  }

return X;
}

double** clip_grad(double** dZ,int r,int c,double range)
{
  for(int i=0;i<r;i++)
  {
      for(int j=0;j<c;j++)
  {
      if(dZ[i][j]<0-range)
      dZ[i][j]=0-range;
      else if(dZ[i][j]>range)
      dZ[i][j]=range;
  }
  }

  return dZ;
}
