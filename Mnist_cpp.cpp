#include<bits/stdc++.h>
#include<random>
#include "Eigen/Dense"
#include <chrono>

using namespace std; 
using namespace Eigen;
#define matrix MatrixXd

struct Parameters{
    matrix W1,B1,W2,B2;
};

struct Gradients{
    matrix dW1,dB1,dW2,dB2;
};

struct Activations{
    matrix Z1,A1,Z2,A2;
};

struct Data{
    matrix x_train,x_test;
    VectorXi y_train,y_test;
};

Parameters initialize(int inputsize,int hiddLayer,int outputLayer){
    matrix W1(hiddLayer,inputsize);
    matrix B1(hiddLayer,1);
    matrix W2(outputLayer,hiddLayer);
    matrix B2(outputLayer,1);

    W1.setRandom();B1.setRandom();W2.setRandom();B2.setRandom();
    W1*=0.5;B1*=0.5;W2*=0.5;B2*=0.5;

    return {W1,B1,W2,B2};
}

matrix ReLU(const matrix &z){
    return (z.array()>0).select(z,0.0);
}

matrix reReLU(const matrix &z){
    return (z.array()>0).cast<double>();
}

matrix softmax(const matrix& z){
    matrix z1=z.rowwise()-z.colwise().maxCoeff();
    z1=z1.array().exp();
    return z1.array().rowwise()/z1.colwise().sum().array();
}

Activations forword_prop(const matrix &x,const Parameters& params){
    matrix Z1=(params.W1*x);
    for(int i=0;i<Z1.cols();i++){
        Z1.col(i)+=params.B1;
    }
    matrix A1=ReLU(Z1);
    matrix Z2=(params.W2*A1);
    for(int i=0;i<Z2.cols();i++){
        Z2.col(i)+=params.B2;
    }
    matrix A2=softmax(Z2);
    return {Z1,A1,Z2,A2};
}

matrix onehot(const VectorXi &y){
    matrix ans=matrix::Zero(10,y.size());
    for(int i=0;i<y.size();i++){
        ans(y(i),i)=1.0;
    }
    return ans;
}

Gradients back_prop(const Activations& acts,const Parameters& params,const matrix& x,const VectorXi &y){
    double n=y.size();
    matrix onehoty=onehot(y);

    matrix dZ2=acts.A2-onehoty;
    matrix dW2=(1/n)*(dZ2*acts.A1.transpose());
    matrix dB2=(1/n)*(dZ2.rowwise().sum());

    matrix dZ1=(params.W2.transpose()*dZ2).array()*reReLU(acts.Z1).array();
    matrix dW1=(1/n)*(dZ1*x.transpose());
    matrix dB1=(1/n)*(dZ1.rowwise().sum());

    return {dW1,dB1,dW2,dB2};
}

Parameters updateparams(Parameters params,const Gradients& grads,double alpha){
    params.W1-=alpha*grads.dW1;
    params.B1-=alpha*grads.dB1;
    params.W2-=alpha*grads.dW2;
    params.B2-=alpha*grads.dB2;
    return params;
}

VectorXi getpredictions(const matrix& a2){
    
    VectorXi predictions(a2.cols());
    for(int i=0;i<a2.cols();i++){
        double maxi=-1.0;
        int maxidx=-1;
        for(int j=0;j<a2.rows();j++){
            if(a2(j,i)>maxi){
                maxi=a2(j,i);
                maxidx=j;
            }
        }
        predictions(i)=maxidx;
    }
    return predictions;
}

double getaccuracy(const VectorXi& predictions, const VectorXi& y){
    return double((predictions.array()==y.array()).count())/double(y.size());
}

Parameters gradient_descent(const matrix&x,const VectorXi&y,double alpha,int iterations){

    Parameters params=initialize(x.rows(),10,10);

    for(int i=0;i<iterations;i++){
        Activations acts=forword_prop(x,params);
        Gradients grads=back_prop(acts,params,x,y);
        params=updateparams(params,grads,alpha);

        if(i%10==0){
            VectorXi predictions=getpredictions(acts.A2);
            cout<<i<<" : "<<fixed<<setprecision(8)<<getaccuracy(predictions,y)<<endl;
        }
    }
    return params;
}

Data loaddata(string filename){
    ifstream file(filename);
    if(!file.is_open())throw runtime_error("Could not open file: "+ filename);

    vector<vector<double>> alldata;
    string line;
    getline(file,line);

    while(getline(file,line)){
        stringstream ss(line);
        string cell;
        vector<double> row;
        while(getline(ss,cell,',')){
            row.push_back(stod(cell));
        }
        alldata.push_back(row);
    }

    auto rnd=default_random_engine{};
    shuffle(alldata.begin(),alldata.end(),rnd);

    int split_index=static_cast<int>(alldata.size()*0.8);
    int num_features=alldata[0].size()-1;

    matrix x_train(num_features,split_index);
    VectorXi y_train(split_index);

    matrix x_test(num_features,alldata.size()-split_index);
    VectorXi y_test(alldata.size()-split_index);

    for(int i=0;i<split_index;i++){
        y_train(i)=alldata[i][0];
        for(int j=0;j<num_features;j++){
            x_train(j,i)=alldata[i][j+1]/255.0;
        }
    }

    for(int i=split_index;i<alldata.size();i++){
        y_test(i-split_index)=alldata[i][0];
        for(int j=0;j<num_features;j++){
            x_test(j,i-split_index)=alldata[i][j+1]/255.0;
        }
    }
    return {x_train,x_test,y_train,y_test};
}

int main(){
    try{
        auto start=chrono::high_resolution_clock::now();

        cout<<"Loading Data..."<<endl;
        Data data=loaddata("train.csv");

        cout<<"x_train :"<<data.x_train.rows()<<" "<<data.x_train.cols()<<endl;
        cout<<"y_train :"<<data.y_train.rows()<<" "<<data.y_train.cols()<<endl;
        cout<<"x_test :"<<data.x_test.rows()<<" "<<data.x_test.cols()<<endl;
        cout<<"y_test :"<<data.y_test.rows()<<" "<<data.y_test.cols()<<endl;

        cout<<"Starting Training..."<<endl;
        Parameters trainedparams=gradient_descent(data.x_train,data.y_train,0.1,500);
        cout<<"Training finished!!"<<endl;

        auto end=chrono::high_resolution_clock::now();
        cout<<"Time took in training = "<<chrono::duration_cast<chrono::seconds>(end-start).count()<<"seconds"<<endl;
        cout<<"Time taken per iteration = "<<chrono::duration_cast<chrono::milliseconds>(end-start).count()/500<<"ms"<<endl;


        Activations test_acts=forword_prop(data.x_test,trainedparams);
        VectorXi testans=getpredictions(test_acts.A2);
        cout<<"Accuracy is = "<<fixed<<setprecision(8)<<getaccuracy(testans,data.y_test)<<endl;

    }catch(exception e){
        cerr<<e.what()<<endl;
        return 1;
    }
    return 0;
}
