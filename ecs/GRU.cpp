#include "GRU.h"


GRU::GRU()
{

}

GRU::~GRU()
{

}

void GRU::setDims(int hidenDims, int trainNums, int predictNums)
{
    hDim = hidenDims;
    uNum = trainNums+predictNums;
    pNum = predictNums;
}

void GRU::setData(vector<vector<double> > &X, vector<vector<double> > &Y, double _step, int _iterateNum, double _targetError)
{
    x = X;
    y = Y;
    xDim = x.size();
    yDim = y.size();
    step = _step;
    iterateNum = _iterateNum;
    targetError = _targetError;
}

// 主体函数，error与matla仿真前6循环一致
void GRU::startTrainning()
{
    scaleX = squrshTo(x,0.0,1.0);
    scaleY = squrshTo(y,0.0,1.0);
    x = matT(x);
    y = matT(y);
    for(int loop=0;loop<iterateNum;loop++)
    {
        // Forward，与matlab仿真前3循环一致

        // rValue[0] = matSigmoidF(x[0]*Wr);
        matMul(x[0],Wr,vTemp_1xh[0]);
        matSigmoidF(vTemp_1xh[0],rValue[0]);

        // hBarValue[0] = matTanhF(x[0]*W);
        matMul(x[0],W,vTemp_1xh[0]);
        matTanhF(vTemp_1xh[0],hBarValue[0]);

        // zValue[0] = matSigmoidF(x[0]*Wz);
        matMul(x[0],Wz,vTemp_1xh[0]);
        matSigmoidF(vTemp_1xh[0],zValue[0]);

        // hValue[0] = matDotMul(zValue[0],hBarValue[0]);
        matDotMul(zValue[0],hBarValue[0],hValue[0]);

        // yValue[0] = matSigmoidF(hValue[0]*Wy);
        matMul(hValue[0],Wy,vTemp_1xy[0]);
        matSigmoidF(vTemp_1xy[0],yValue[0]);

        for(int t=1;t<uNum-pNum;t++)
        {
            // rValue[t] = matSigmoidF(x[t]*Wr+hValue[t-1]*Ur);
            matMul(x[t],Wr,vTemp_1xh[0]);
            matMul(hValue[t-1],Ur,vTemp_1xh[1]);
            matAdd(vTemp_1xh[0],vTemp_1xh[1],vTemp_1xh[0]);
            matSigmoidF(vTemp_1xh[0],rValue[t]);

            // hBarValue[t] = matTanhF(x[t]*W+(matDotMul(rValue[t],hValue[t-1]))*U);
            matDotMul(rValue[t],hValue[t-1],vTemp_1xh[0]);
            matMul(vTemp_1xh[0],U,vTemp_1xh[1]);
            matMul(x[t],W,vTemp_1xh[2]);
            matAdd(vTemp_1xh[1],vTemp_1xh[2],vTemp_1xh[0]);
            matTanhF(vTemp_1xh[0],hBarValue[t]);

            // zValue[t] = matSigmoidF(x[t]*Wz+hValue[t-1]*Uz);
            matMul(x[t],Wz,vTemp_1xh[0]);
            matMul(hValue[t-1],Uz,vTemp_1xh[1]);
            matAdd(vTemp_1xh[0],vTemp_1xh[1],vTemp_1xh[0]);
            matSigmoidF(vTemp_1xh[0],zValue[t]);

            // hValue[t] = matDotMul(1-zValue[t],hValue[t-1])+matDotMul(zValue[t],hBarValue[t]);
            matSub(1,zValue[t],vTemp_1xh[0]);
            matDotMul(vTemp_1xh[0],hValue[t-1],vTemp_1xh[0]);
            matDotMul(zValue[t],hBarValue[t],vTemp_1xh[1]);
            matAdd(vTemp_1xh[0],vTemp_1xh[1],hValue[t]);

            // yValue[t] = matSigmoidF(hValue[t]*Wy);
            matMul(hValue[t],Wy,vTemp_1xy[0]);
            matSigmoidF(vTemp_1xy[0],yValue[t]);
        }

        // Backward，与matlab仿真前3循环一致
        clearBackwardTempValues();
        for(int t=uNum-pNum-1;t>=1;t--)
        {
            // delta_y = matDotMul(yValue[t]-y[t],matSigmoidB(yValue[t]));
            matSub(yValue[t],y[t],vTemp_1xy[0]);
            matSigmoidB(yValue[t],vTemp_1xy[1]);
            matDotMul(vTemp_1xy[0],vTemp_1xy[1],delta_y);

            // delta_h = delta_y*matT(Wy) + delta_z_Next*matT(Uz) + matDotMul(delta_Next*matT(U),rValue[t+1]) +
            //        delta_r_Next*matT(Ur) + matDotMul(delta_h_Next,1-zValue[t+1]);
            matT(Wy,mTemp_yxh_1);
            matMul(delta_y,mTemp_yxh_1,vTemp_1xh[0]);
            matT(Uz,mTemp_hxh_1);
            matMul(delta_z_Next,mTemp_hxh_1,vTemp_1xh[1]);
            matT(Ur,mTemp_hxh_1);
            matMul(delta_r_Next,mTemp_hxh_1,vTemp_1xh[2]);
            matT(U,mTemp_hxh_1); // mTemp_hxh_1当前域为U的转置
            matMul(delta_Next,mTemp_hxh_1,vTemp_1xh[3]);
            matDotMul(vTemp_1xh[3],rValue[t+1],vTemp_1xh[3]);
            matSub(1,zValue[t+1],vTemp_1xh[4]);
            matDotMul(delta_h_Next,vTemp_1xh[4],vTemp_1xh[4]);
            matAdd(vTemp_1xh,5,delta_h);

            // delta_z = matDotMul(delta_h,hBarValue[t]-hValue[t-1],matSigmoidB(zValue[t]));
            matSub(hBarValue[t],hValue[t-1],vTemp_1xh[0]);
            matSigmoidB(zValue[t],vTemp_1xh[1]);
            ptr_vTemp_1xh[0] = &vTemp_1xh[0];
            ptr_vTemp_1xh[1] = &vTemp_1xh[1];
            ptr_vTemp_1xh[2] = &delta_h;
            matDotMul(ptr_vTemp_1xh,3,delta_z);

            // delta   = matDotMul(delta_h,zValue[t],matTanhB(hBarValue[t]));
            matTanhB(hBarValue[t],vTemp_1xh[0]); // vTemp_1xh[0]当前为matTanhB
            ptr_vTemp_1xh[0] = &delta_h;
            ptr_vTemp_1xh[1] = &zValue[t];
            ptr_vTemp_1xh[2] = &vTemp_1xh[0];
            matDotMul(ptr_vTemp_1xh,3,delta);

            // delta_r = matDotMul(hValue[t-1],
            //        (matDotMul(delta_h,zValue[t],matTanhB(hBarValue[t])))*matT(U),
            //        matSigmoidB(rValue[t]));
            ptr_vTemp_1xh[0] = &delta_h;
            ptr_vTemp_1xh[1] = &zValue[t];
            ptr_vTemp_1xh[2] = &vTemp_1xh[0];
            matDotMul(ptr_vTemp_1xh,3,vTemp_1xh[3]);
            matMul(vTemp_1xh[3],mTemp_hxh_1,vTemp_1xh[4]);
            matSigmoidB(rValue[t],vTemp_1xh[3]);
            ptr_vTemp_1xh[0] = &hValue[t-1];
            ptr_vTemp_1xh[1] = &vTemp_1xh[4];
            ptr_vTemp_1xh[2] = &vTemp_1xh[3];
            matDotMul(ptr_vTemp_1xh,3,delta_r);

            // 减少转换次数
            matT(hValue[t],vTemp_hx1_1);
            matT(hValue[t-1],vTemp_hx1_2);
            matT(x[t],vTemp_xx1);
            // dWy = dWy+matT(hValue[t])*delta_y;
            matMul(vTemp_hx1_1,delta_y,mTemp_hxy_1); // vTemp_hx1_1可以被占用
            matAdd(dWy,mTemp_hxy_1,dWy);

            // dWz = dWz+matT(x[t])*delta_z;
            matMul(vTemp_xx1,delta_z,mTemp_xxh_1);
            matAdd(dWz,mTemp_xxh_1,dWz);

            // dUz = dUz+matT(hValue[t-1])*delta_z;
            matMul(vTemp_hx1_2,delta_z,mTemp_hxh_1);
            matAdd(dUz,mTemp_hxh_1,dUz);

            // dW  = dW+matT(x[t])*delta;
            matMul(vTemp_xx1,delta,mTemp_xxh_1);
            matAdd(dW,mTemp_xxh_1,dW);

            // dU  = dU+matT(matDotMul(rValue[t],hValue[t-1]))*delta;
            matDotMul(rValue[t],hValue[t-1],vTemp_1xh[0]);
            matT(vTemp_1xh[0],vTemp_hx1_1);
            matMul(vTemp_hx1_1,delta,mTemp_hxh_1);
            matAdd(dU,mTemp_hxh_1,dU);

            // dWr = dWr+matT(x[t])*delta_r;
            matMul(vTemp_xx1,delta_r,mTemp_xxh_1);
            matAdd(dWr,mTemp_xxh_1,dWr);

            // dUr = dUr+matT(hValue[t-1])*delta_r;
            matMul(vTemp_hx1_2,delta_r,mTemp_hxh_1);
            matAdd(dUr,mTemp_hxh_1,dUr);

            delta_r_Next = delta_r;
            delta_z_Next = delta_z;
            delta_h_Next = delta_h;
            delta_Next   = delta;
        }

        int t =0;
        // delta_y = matDotMul(yValue[t]-y[t],matSigmoidB(yValue[t]));
        matSub(yValue[t],y[t],vTemp_1xy[0]);
        matSigmoidB(yValue[t],vTemp_1xy[1]);
        matDotMul(vTemp_1xy[0],vTemp_1xy[1],delta_y);

        // delta_h = delta_y*matT(Wy) + delta_z_Next*matT(Uz) + matDotMul(delta_Next*matT(U),rValue[t+1]) +
        //          delta_r_Next*matT(Ur) + matDotMul(delta_h_Next,1-zValue[t+1]);
        matT(Wy,mTemp_yxh_1);
        matMul(delta_y,mTemp_yxh_1,vTemp_1xh[0]);
        matT(Uz,mTemp_hxh_1);
        matMul(delta_z_Next,mTemp_hxh_1,vTemp_1xh[1]);
        matT(Ur,mTemp_hxh_1);
        matMul(delta_r_Next,mTemp_hxh_1,vTemp_1xh[2]);
        matT(U,mTemp_hxh_1); // mTemp_hxh_1当前域为U的转置
        matMul(delta_Next,mTemp_hxh_1,vTemp_1xh[3]);
        matDotMul(vTemp_1xh[3],rValue[t+1],vTemp_1xh[3]);
        matSub(1,zValue[t+1],vTemp_1xh[4]);
        matDotMul(delta_h_Next,vTemp_1xh[4],vTemp_1xh[4]);
        matAdd(vTemp_1xh,5,delta_h);

        // delta_z = matDotMul(delta_h,hBarValue[t],matSigmoidB(zValue[t]));
        matSigmoidB(zValue[t],vTemp_1xh[0]);
        ptr_vTemp_1xh[0] = &delta_h;
        ptr_vTemp_1xh[1] = &hBarValue[t];
        ptr_vTemp_1xh[2] = &vTemp_1xh[0];
        matDotMul(ptr_vTemp_1xh,3,delta_z);

        // delta   = matDotMul(delta_h,zValue[t],matTanhB(hBarValue[t]));
        matTanhB(hBarValue[t],vTemp_1xh[0]); // vTemp_1xh[0]当前为matTanhB
        ptr_vTemp_1xh[0] = &delta_h;
        ptr_vTemp_1xh[1] = &zValue[t];
        ptr_vTemp_1xh[2] = &vTemp_1xh[0];
        matDotMul(ptr_vTemp_1xh,3,delta);

        delta_r.assign(delta_r.size(),0.0);

        // 减少转换次数
        matT(hValue[t],vTemp_hx1_1);
        matT(x[t],vTemp_xx1);

         // dWy = dWy+matT(hValue[t])*delta_y;
        matMul(vTemp_hx1_1,delta_y,mTemp_hxy_1); // vTemp_hx1_1可以被占用
        matAdd(dWy,mTemp_hxy_1,dWy);

         // dWz = dWz+matT(x[t])*delta_z;
        matMul(vTemp_xx1,delta_z,mTemp_xxh_1);
        matAdd(dWz,mTemp_xxh_1,dWz);

         // dW  = dW+matT(x[t])*delta;
        matMul(vTemp_xx1,delta,mTemp_xxh_1);
        matAdd(dW,mTemp_xxh_1,dW);

         // dWr = dWr+matT(x[t])*delta_r;
         matMul(vTemp_xx1,delta_r,mTemp_xxh_1);
         matAdd(dWr,mTemp_xxh_1,dWr);

         // Fix
         // Wy = Wy-step*dWy;
         matDotMul(step,dWy,mTemp_hxy_1);
         matSub(Wy,mTemp_hxy_1,Wy);

         // Wr = Wr-step*dWr;
         matDotMul(step,dWr,mTemp_xxh_1);
         matSub(Wr,mTemp_xxh_1,Wr);

         // Ur = Ur-step*dUr;
         matDotMul(step,dUr,mTemp_hxh_1);
         matSub(Ur,mTemp_hxh_1,Ur);

         // W  = W -step*dW;
         matDotMul(step,dW,mTemp_xxh_1);
         matSub(W,mTemp_xxh_1,W);

         // U  = U -step*dU;
         matDotMul(step,dU,mTemp_hxh_1);
         matSub(U,mTemp_hxh_1,U);

         // Wz = Wz-step*dWz;
         matDotMul(step,dWz,mTemp_xxh_1);
         matSub(Wz,mTemp_xxh_1,Wz);

         // Uz = Uz-step*dUz;
         matDotMul(step,dUz,mTemp_hxh_1);
         matSub(Uz,mTemp_hxh_1,Uz);

         // 存储error最小下的参数
         if(loop == 0)
         {
           error = minError = getError(yValue,y);
           store[0] = Wy;
           store[1] = Wr;
           store[2] = Ur;
           store[3] = W;
           store[4] = U;
           store[5] = Wz;
           store[6] = Uz;
         }
         else
         {
             error = getError(yValue,y);
             if(error < minError)
             {
                 minError = error;
                 store[0] = Wy;
                 store[1] = Wr;
                 store[2] = Ur;
                 store[3] = W;
                 store[4] = U;
                 store[5] = Wz;
                 store[6] = Uz;
             }
         }
         if(loop % 500 == 0)
//             cout << "Error in loop " << loop << " : " << error << endl;
         if(error < targetError)
             break;
    }

    // predict
    Wy = store[0];
    Wr = store[1];
    Ur = store[2];
    W  = store[3];
    U  = store[4];
    Wz = store[5];
    Uz = store[6];
    // rValue[0] = matSigmoidF(x[0]*Wr);
    matMul(x[0],Wr,vTemp_1xh[0]);
    matSigmoidF(vTemp_1xh[0],rValue[0]);

    // hBarValue[0] = matTanhF(x[0]*W);
    matMul(x[0],W,vTemp_1xh[0]);
    matTanhF(vTemp_1xh[0],hBarValue[0]);

    // zValue[0] = matSigmoidF(x[0]*Wz);
    matMul(x[0],Wz,vTemp_1xh[0]);
    matSigmoidF(vTemp_1xh[0],zValue[0]);

    // hValue[0] = matDotMul(zValue[0],hBarValue[0]);
    matDotMul(zValue[0],hBarValue[0],hValue[0]);

    // yValue[0] = matSigmoidF(hValue[0]*Wy);
    matMul(hValue[0],Wy,vTemp_1xy[0]);
    matSigmoidF(vTemp_1xy[0],yValue[0]);

    for(int t=1;t<uNum;t++)
    {
        // rValue[t] = matSigmoidF(x[t]*Wr+hValue[t-1]*Ur);
        matMul(x[t],Wr,vTemp_1xh[0]);
        matMul(hValue[t-1],Ur,vTemp_1xh[1]);
        matAdd(vTemp_1xh[0],vTemp_1xh[1],vTemp_1xh[0]);
        matSigmoidF(vTemp_1xh[0],rValue[t]);

        // hBarValue[t] = matTanhF(x[t]*W+(matDotMul(rValue[t],hValue[t-1]))*U);
        matDotMul(rValue[t],hValue[t-1],vTemp_1xh[0]);
        matMul(vTemp_1xh[0],U,vTemp_1xh[1]);
        matMul(x[t],W,vTemp_1xh[2]);
        matAdd(vTemp_1xh[1],vTemp_1xh[2],vTemp_1xh[0]);
        matTanhF(vTemp_1xh[0],hBarValue[t]);

        // zValue[t] = matSigmoidF(x[t]*Wz+hValue[t-1]*Uz);
        matMul(x[t],Wz,vTemp_1xh[0]);
        matMul(hValue[t-1],Uz,vTemp_1xh[1]);
        matAdd(vTemp_1xh[0],vTemp_1xh[1],vTemp_1xh[0]);
        matSigmoidF(vTemp_1xh[0],zValue[t]);

        // hValue[t] = matDotMul(1-zValue[t],hValue[t-1])+matDotMul(zValue[t],hBarValue[t]);
        matSub(1,zValue[t],vTemp_1xh[0]);
        matDotMul(vTemp_1xh[0],hValue[t-1],vTemp_1xh[0]);
        matDotMul(zValue[t],hBarValue[t],vTemp_1xh[1]);
        matAdd(vTemp_1xh[0],vTemp_1xh[1],hValue[t]);

        // yValue[t] = matSigmoidF(hValue[t]*Wy);
        matMul(hValue[t],Wy,vTemp_1xy[0]);
        matSigmoidF(vTemp_1xy[0],yValue[t]);
    }

    // 恢复压缩的输出
    cout << "Min Error is " << minError << endl;
    matDotMul(scaleY,yValue,yValue);
}

double GRU::sigmoidForward(double x)
{
    return 1.0/(1.0+exp(-x));
}

double GRU::sigmoidBackward(double x)
{
    return x*(1.0-x);
}

void GRU::matSigmoidF(const vector<double> &vec, vector<double> &vecOutput)
{
    if(vec.size() != vecOutput.size())
    {
        cout << "Size error in matSigmoidF, input: vec; output: vecOutput." << endl;
        exit(-1);
        return;
    }
    else
    {
        for(uint i=0;i<vec.size();i++)
            vecOutput[i] = sigmoidForward(vec[i]);
    }
}

void GRU::matSigmoidB(const vector<double> &vec, vector<double> &vecOutput)
{
    if(vec.size() != vecOutput.size())
    {
        cout << "Size error in matSigmoidB, input: vec; output: vecOutput." << endl;
        exit(-1);
        return;
    }
    else
    {
        for(uint i=0;i<vec.size();i++)
            vecOutput[i] = sigmoidBackward(vec[i]);
    }
}

double GRU::tanhForward(double x)
{
    return 2.0/(1.0+exp(-2.0*x))-1.0;
}

double GRU::tanhBackward(double x)
{
    return 1.0-pow(x,2);
}

void GRU::matTanhF(const vector<double> &vec, vector<double> &vecOutput)
{
    if(vec.size() != vecOutput.size())
    {
        cout << "Size error in matTanF, input: vec; output: vec." << endl;
        exit(-1);
        return;
    }
    {
        for(uint i=0;i<vec.size();i++)
            vecOutput[i] = tanhForward(vec[i]);
    }
}

void GRU::matTanhB(const vector<double> &vec, vector<double> &vecOutput)
{
    if(vec.size() != vecOutput.size())
    {
        cout << "Size error in matTanF, input: vec; output: vec." << endl;
        exit(-1);
        return;
    }
    {
        for(uint i=0;i<vec.size();i++)
            vecOutput[i] = tanhBackward(vec[i]);
    }
}

void GRU::getPredictArray(vector<double> &output)
{
    if(output.size() != yValue.size())
    {
        cout << "Error in return predict Y!" << endl;
        return;
    }
    for(uint i=0;i<yValue.size();i++)
        output[i] = yValue[i][0];
}

// 分配空间
void GRU::initCell()
{
    store.resize(7);
    Wy.resize(hDim);
    dWy.resize(hDim);
    for(int i=0;i<hDim;i++)
    {
        Wy[i].resize(yDim);
        dWy[i].resize(yDim);
    }
    Wr.resize(xDim);
    dWr.resize(xDim);
    W.resize(xDim);
    dW.resize(xDim);
    Wz.resize(xDim);
    dWz.resize(xDim);
    for(int i=0;i<xDim;i++)
    {
        Wr[i].resize(hDim);
        dWr[i].resize(hDim);
        W[i].resize(hDim);
        dW[i].resize(hDim);
        Wz[i].resize(hDim);
        dWz[i].resize(hDim);
    }
    Ur.resize(hDim);
    dUr.resize(hDim);
    U.resize(hDim);
    dU.resize(hDim);
    Uz.resize(hDim);
    dUz.resize(hDim);
    for(int i=0;i<hDim;i++)
    {
        Ur[i].resize(hDim);
        dUr[i].resize(hDim);
        U[i].resize(hDim);
        dU[i].resize(hDim);
        Uz[i].resize(hDim);
        dUz[i].resize(hDim);
    }

    rValue.resize(uNum+1);
    for(int i=0;i<uNum+1;i++)
        rValue[i].resize(hDim);
    zValue.resize(uNum+1);
    for(int i=0;i<uNum+1;i++)
        zValue[i].resize(hDim);
    hBarValue.resize(uNum);
    for(int i=0;i<uNum;i++)
        hBarValue[i].resize(hDim);
    hValue.resize(uNum);
    for(int i=0;i<uNum;i++)
        hValue[i].resize(hDim);
    yValue.resize(uNum);
    for(int i=0;i<uNum;i++)
        yValue[i].resize(yDim);

    delta_r_Next.resize(hDim);
    delta_z_Next.resize(hDim);
    delta_h_Next.resize(hDim);
    delta_Next.resize(hDim);

    delta_y.resize(yDim);
    delta_h.resize(hDim);
    delta_z.resize(hDim);
    delta.resize(hDim);
    delta_r.resize(hDim);

    // Temp vector
    vTemp_1xh.resize(5);
    for(int i=0;i<5;i++)
        vTemp_1xh[i].resize(hDim);
    vTemp_1xy.resize(2);
    for(int i=0;i<2;i++)
        vTemp_1xy[i].resize(yDim);

    mTemp_yxh_1.resize(yDim);
    for(int i=0;i<yDim;i++)
        mTemp_yxh_1[i].resize(hDim);
    mTemp_hxh_1.resize(hDim);
    for(int i=0;i<hDim;i++)
        mTemp_hxh_1[i].resize(hDim);
     ptr_vTemp_1xh.resize(5);
     vTemp_hx1_1.resize(hDim);
     vTemp_hx1_2.resize(hDim);
     for(uint i=0;i<vTemp_hx1_1.size();i++)
     {
         vTemp_hx1_1[i].resize(1);
         vTemp_hx1_2[i].resize(1);
     }
     vTemp_xx1.resize(xDim);
     for(uint i=0;i<vTemp_xx1.size();i++)
         vTemp_xx1[i].resize(1);
     mTemp_hxy_1.resize(hDim);
     for(uint i=0;i<mTemp_hxy_1.size();i++)
         mTemp_hxy_1[i].resize(yDim);
     mTemp_xxh_1.resize(xDim);
     for(uint i=0;i<mTemp_xxh_1.size();i++)
         mTemp_xxh_1[i].resize(hDim);

}

// 初始化值
void GRU::initCellValue()
{
    fillWithRandomValue(Wy,-1.0,1.0);
    fillWithRandomValue(Ur,-1.0,1.0);
    fillWithRandomValue(U,-1.0,1.0);
    fillWithRandomValue(Uz,-1.0,1.0);
    fillWithRandomValue(Wr,-1.0,1.0);
    fillWithRandomValue(W,-1.0,1.0);
    fillWithRandomValue(Wz,-1.0,1.0);

    for(uint i=0;i<rValue.size();i++)
    {
        rValue[i].assign(rValue[0].size(),0.0);
        zValue[i].assign(rValue[0].size(),0.0);
    }
    for(uint i=0;i<hBarValue.size();i++)
    {
        hBarValue[i].assign(hBarValue[0].size(),0.0);
        hValue[i].assign(hBarValue[0].size(),0.0);
    }
    for(uint i=0;i<yValue.size();i++)
    {
        yValue[i].assign(yValue[0].size(),0.0);
    }
    delta_r_Next.assign(hDim,0.0);
    delta_z_Next.assign(hDim,0.0);
    delta_h_Next.assign(hDim,0.0);
    delta_Next.assign(hDim,0.0);
}

// 填充[-1,1]随机数
void GRU::fillWithRandomValue(vector<vector<double> > &mat, const double &a, const double &b)
{
    default_random_engine dre;
    dre.seed(time(0));
    uniform_real_distribution<double> dr(a,b);
    for(uint i=0;i<mat.size();i++)
        for(uint j=0;j<mat[0].size();j++)
            mat[i][j] = dr(dre);
}

void GRU::fillWithRandomValue(vector<double> &vec, const double &a, const double &b)
{
    default_random_engine dre;
    dre.seed(time(0));
    uniform_real_distribution<double> dr(a,b);
    for(uint i=0;i<vec.size();i++)
        vec[i] = dr(dre);
}

// 计算误差，未单元测试
double GRU::getError(vector<vector<double> > &fit, vector<vector<double> > &target)
{
    double output = 0.0;
    double temp;
    if(fit.size() != target.size() || fit[0].size() != target[0].size())
    {
        cout << " Error in getError!";
        return -1;
    }
    else
    {
        for(uint i=0;i<target.size();i++)
        {
            for(uint j=0;j<target[0].size();j++)
            {
                temp = fit[i][j]-target[i][j];
                output += pow(temp,2);
            }
        }
        output = sqrt(output)/double(yDim);
        return output;
    }
}

// 线性压缩数据到[a,b]之间
double GRU::squrshTo(vector<vector<double>> &src, double a, double b)
{
    double MAX_VALUE = src[0][0];
    double n = b-a;
    for(uint i=0;i<src.size();i++)
    {
        for(uint j=0;j<src[0].size();j++)
        {
            if(MAX_VALUE < src[i][j])
                MAX_VALUE = src[i][j];
        }
    }
    if(MAX_VALUE == 0)
        MAX_VALUE = 1;
    for(uint i=0;i<src.size();i++)
    {
        for(uint j=0;j<src[0].size();j++)
        {
            src[i][j] = src[i][j]/MAX_VALUE*n+a;
        }
    }
    return MAX_VALUE/n;
}

void GRU::clearBackwardTempValues()
{
    delta_r_Next.assign(delta_r_Next.size(),0.0);
    delta_r_Next.assign(delta_z_Next.size(),0.0);
    delta_h_Next.assign(delta_h_Next.size(),0.0);
    delta_Next.assign(delta_Next.size(),0.0);
    for(uint i=0;i<dWy.size();i++)
    {
        dWy[i].assign(dWy[0].size(),0.0);
    }
    for(uint i=0;i<dUr.size();i++)
    {
        dUr[i].assign(dUr[0].size(),0.0);
        dU[i].assign(dU[0].size(),0.0);
        dUz[i].assign(dUz[0].size(),0.0);
    }
    for(uint i=0;i<dWr.size();i++)
    {
        dWr[i].assign(dWr[0].size(),0.0);
        dW[i].assign(dW[0].size(),0.0);
        dWz[i].assign(dWz[0].size(),0.0);
    }
}

// 矩阵转置，采用return返回，效率较低，只进行过一次单元测试
vector<vector<double> > matT(const vector<vector<double> > &src)
{
    vector<vector<double>> dst(src[0].size());
    for(uint i=0;i<src[0].size();i++)
        dst[i].resize(src.size());
    for(uint i=0;i<src.size();i++)
    {
        for(uint j=0;j<src[0].size();j++)
            dst[j][i] = src[i][j];
    }
    return dst;
}

// 向量转置，未单元测试
vector<vector<double> > matT(const vector<double> &src)
{
    vector<vector<double>> dst(src.size());
    for(uint i=0;i<src.size();i++)
        dst[i].resize(1);
    for(uint i=0;i<src.size();i++)
    {
            dst[i][0] = src[i];
    }
    return dst;
}

// 向量转置，未单元测试
vector<double> matT(const vector<vector<double> > &src, int type)
{
    vector<double> dst(src.size());
    for(uint i=0;i<src.size();i++)
    {
            dst[i] = src[i][0];
    }
    return dst;
}

// 向量乘矩阵输出向量，未单元测试
void matMul(const vector<double> &vec, const vector<vector<double> > &mat, vector<double> &vecOutput)
{
    if(vec.size() != mat.size() || mat[0].size() != vecOutput.size())
    {
        cout << "Size error in matrixMul, input: vec, mat; output: vec." << endl;
        exit(-1);
        return;
    }
    else if(&vec == &vecOutput)
    {
        cout << "Error in matrixMul, input and output have same address." << endl;
        exit(-1);
        return;
    }
    else
    {
        vecOutput.assign(vecOutput.size(),0.0);
        for(uint i=0;i<mat[0].size();i++)
        {
            for(uint j=0;j<vec.size();j++)
            {
                vecOutput[i] += (vec[j]*mat[j][i]);
            }
        }
    }
}

// 向量与向量点乘，未单元测试
void matDotMul(const vector<double> &vec1, const vector<double> &vec2, vector<double> &vecOutput)
{
    if(vec1.size() != vec2.size() || vec1.size() != vecOutput.size())
    {
        cout << "Size error in  matDotMul, input: vec, vec; output: vec." << endl;
        exit(-1);
        return;
    }
    else
    {
        for(uint i=0;i<vec1.size();i++)
            vecOutput[i] = vec1[i]*vec2[i];
    }
}

// 向量与向量相加，未单元测试
void matAdd(const vector<double> &vec1, const vector<double> &vec2, vector<double> &vecOutput)
{
    if(vec1.size() != vec2.size() || vec1.size() != vecOutput.size())
    {
        cout << "Size error in matAdd, input: vec, vec; output: vec." << endl;
        exit(-1);
        return;
    }
    else
    {
        for(uint i=0;i<vec1.size();i++)
            vecOutput[i] = vec1[i]+vec2[i];
    }
}

// 常数与向量相减，未单元测试
void matSub(double a, const vector<double> &vec, vector<double> &vecOutput)
{
    if(vec.size() != vecOutput.size())
    {
        cout << "Size error in matSub, input: double, vec; output: vec." << endl;
        exit(-1);
        return;
    }
    else
    {
        for(uint i=0;i<vec.size();i++)
            vecOutput[i] = a-vec[i];
    }
}

// 向量相减，未单元测试
void matSub(const vector<double> &vec1, const vector<double> &vec2, vector<double> &vecOutput)
{
    if(vec1.size() != vec2.size() || vec1.size() != vecOutput.size())
    {
        cout << "Size error in matSub, input: vec, vec; output: vec." << endl;
        exit(-1);
        return;
    }
    else
    {
        for(uint i=0;i<vec1.size();i++)
            vecOutput[i] = vec1[i]-vec2[i];
    }
}

// 矩阵转置，未单元测试
void matT(const vector<vector<double> > &mat, vector<vector<double> > &matOutput)
{
    if(mat.size() != matOutput[0].size() || mat[0].size() != matOutput.size())
    {
        cout << "Size error in matT, input: mat; output: mat." << endl;
        exit(-1);
        return;
    }
    {
        for(uint i=0;i<mat.size();i++)
        {
            for(uint j=0;j<mat[0].size();j++)
                matOutput[j][i] = mat[i][j];
        }
    }
}

// 多向量相加，未单元测试
void matAdd(const vector<vector<double> > &vecPack, int packSize, vector<double> &vecOutput)
{
    for(int i=0;i<packSize;i++)
    {
        if(&vecPack[i] == &vecOutput)
        {
            cout << "Error in matAdd, input and output have same address." << endl;
            exit(-1);
            return;
        }
        if(vecPack[0].size() != vecOutput.size())
        {
            cout << "Size error in matAdd, input: vecPack; output: vec." << endl;
            exit(-1);
            return;
        }
    }
    for(uint i=0;i<vecPack[0].size();i++)
    {
        vecOutput[i] = 0.0;
        for(int j=0;j<packSize;j++)
        {
            vecOutput[i] += vecPack[j][i];
        }
    }
}

// 多向量点乘，未单元测试
void matDotMul(const vector<vector<double> > &vecPack, int packSize, vector<double> &vecOutput)
{
    for(int i=0;i<packSize;i++)
    {
        if(&vecPack[i] == &vecOutput)
        {
            cout << "Error in matDotMul, input and output have same address." << endl;
            exit(-1);
            return;
        }
        if(vecPack[0].size() != vecOutput.size())
        {
            cout << "Size error in matDotMul, input: vecPack; output: vec." << endl;
            exit(-1);
            return;
        }
    }
    for(uint i=0;i<vecPack[0].size();i++)
    {
        vecOutput[i] = 1.0;
        for(int j=0;j<packSize;j++)
        {
            vecOutput[i] *= vecPack[j][i];
        }
    }
}

// 多向量点乘，未单元测试
void matDotMul(const vector<vector<double> *> vecPack, int packSize, vector<double> &vecOutput)
{
    for(int i=0;i<packSize;i++)
    {
        if(vecPack[i] == &vecOutput)
        {
            cout << "Error in matMul, input and output have same address." << endl;
            exit(-1);
            return;
        }
        if(vecPack[0]->size() != vecOutput.size())
        {
            cout << "Size error in matMul, input: vecPack; output: vec." << endl;
            exit(-1);
            return;
        }
    }
    for(uint i=0;i<vecPack[0]->size();i++)
    {
        vecOutput[i] = 1.0;
        for(int j=0;j<packSize;j++)
        {
            vecOutput[i] *= (*vecPack[j])[i];
        }
    }
}

// 行向量转置成列向量，未单元测试
void matT(const vector<double> &vec, vector<vector<double> > &vecOutput)
{
    if(vec.size() != vecOutput.size())
    {
        cout << "Size error in matT, input: vec; output: vecT." << endl;
        exit(-1);
        return;
    }
    else
    {
        for(uint i=0;i<vec.size();i++)
        {
                vecOutput[i][0] = vec[i];
        }
    }
}

// 矩阵相加，未单元测试
void matAdd(const vector<vector<double> > &mat1, const vector<vector<double> > &mat2, vector<vector<double> > &matOutput)
{
    if(mat1.size() != mat2.size() || mat1.size() != matOutput.size() ||
            mat1[0].size() != mat2[0].size() || mat1[0].size() != matOutput[0].size())
    {
        cout << "Size error in matAdd, input: mat, mat; output: mat." << endl;
        exit(-1);
        return;
    }
    else
    {
        for(uint i=0;i<mat1.size();i++)
        {
            for(uint j=0;j<mat1[0].size();j++)
            {
                matOutput[i][j] = mat1[i][j]+mat2[i][j];
            }
        }
    }
}

// 列向量和行向量相乘，未单元测试
void matMul(const vector<vector<double> > &mat, const vector<double> &vec, vector<vector<double> > &matOutput)
{
    if(mat.size() != matOutput.size() || vec.size() != matOutput[0].size())
    {
        cout << "Size error in matMul, input: vecT, vec; output: mat." << endl;
        exit(-1);
        return;
    }
    else
    {
        for(uint i=0;i<mat.size();i++)
            for(uint j=0;j<vec.size();j++)
                matOutput[i][j] = mat[i][0]*vec[j];
    }
}

// 常数和矩阵点乘，未单元测试
void matDotMul(double a, const vector<vector<double> > &mat, vector<vector<double> > &matOutput)
{
    if(mat.size() != matOutput.size() || mat[0].size() != matOutput[0].size())
    {
        cout << "Size error in matDotMul, input: double, mat; output: mat." << endl;
        exit(-1);
        return;
    }
    else
    {
        for(uint i=0;i<mat.size();i++)
            for(uint j=0;j<mat[0].size();j++)
                matOutput[i][j] = a*mat[i][j];
    }
}

// 矩阵和矩阵相减，未单元测试
void matSub(const vector<vector<double> > &mat1, const vector<vector<double> > &mat2, vector<vector<double> > &matOutput)
{
    if(mat1.size() != mat2.size() || mat1.size() != matOutput.size() ||
       mat1[0].size() != mat2[0].size() || mat1[0].size() != matOutput[0].size())
    {
        cout << "Size error in matSub, input: mat, mat; output: mat." << endl;
        exit(-1);
        return;
    }
    else
    {
        for(uint i=0;i<mat1.size();i++)
        {
            for(uint j=0;j<mat1[0].size();j++)
            {
                matOutput[i][j] = mat1[i][j]-mat2[i][j];
            }
        }
    }
}
