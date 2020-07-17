using namespace std;

class LLE
{
    double** X;
    double** Z;
    double** W;
    int in_dims;
    int batch_size;
    int out_dims;
    double loss_w;
    double loss_z;
   

    public :
    LLE(double** X,int in_dims,int out_dims,int batch_size):
    X(X),in_dims(in_dims),out_dims(out_dims),batch_size(batch_size)
    {
        W=new double*[batch_size];
        for(int i=0;i<batch_size;i++)
        {
            W[i]=new double[batch_size];
            for(int j=0;j<batch_size;j++)
            W[i][j]=rand()/(1.0+RAND_MAX);
        }
        Z=new double*[batch_size];
        for(int i=0;i<batch_size;i++)
        {
        Z[i]=new double[out_dims];
        for(int j=0;j<out_dims;j++)
        Z[i][j]=20*(rand()/(1.0+RAND_MAX));
        }
    }
    void update_weights(int epochs,double lr)
    {
        for(int i=0;i<epochs;i++)
        {
            
            double** temp=mat_sub(X,mat_mul(W,X,batch_size,batch_size,batch_size,in_dims),batch_size,in_dims,batch_size,in_dims);
            this->loss_w=(mat_sum(mat_sum(mat_ele_mul(temp,temp,batch_size,in_dims,batch_size,in_dims),batch_size,in_dims,1),batch_size,1,0)[0][0])/(1.0*batch_size);
            cout<<"Loss W : "<<loss_w<<endl;
            temp=mat_mul(W,X,batch_size,batch_size,batch_size,in_dims);
            temp=mat_sub(temp,X,batch_size,in_dims,batch_size,in_dims);
            temp=mat_transpose(temp,batch_size,in_dims);
            temp=mat_mul(X,temp,batch_size,in_dims,in_dims,batch_size);
            double** dW=mat_ele_mul(lr*2/batch_size,temp,batch_size,batch_size);
            W=mat_sub(W,dW,batch_size,batch_size,batch_size,batch_size);
        }
    }
    void update_z(int epochs,double lr,double C=1e-5)
    {
        for(int i=0;i<epochs;i++)
        {
        double** temp1=mat_divide(C,Z,batch_size,out_dims);
        double** temp=mat_sub(Z,mat_mul(W,Z,batch_size,batch_size,batch_size,out_dims),batch_size,out_dims,batch_size,out_dims);
        temp=mat_add(temp1,temp,batch_size,out_dims,batch_size,out_dims);
        this->loss_z=(mat_sum(mat_sum(mat_ele_mul(temp,temp,batch_size,out_dims,batch_size,out_dims),batch_size,out_dims,1),batch_size,1,0)[0][0])/(1.0*batch_size);
        cout<<"Loss Z :"<<loss_z<<endl;

        
        temp1=mat_divide(-(C),mat_ele_mul(Z,Z,batch_size,out_dims,batch_size,out_dims),batch_size,out_dims);
        temp=mat_add(temp,temp1,batch_size,out_dims,batch_size,out_dims);
        temp=mat_ele_mul(lr*2/batch_size,temp,batch_size,out_dims);
        Z=mat_add(Z,temp,batch_size,out_dims,batch_size,out_dims);
        }
    }


    double** get_W()
    {
        return W;
    }
    double** get_Z()
    {
        return Z;
    }


};