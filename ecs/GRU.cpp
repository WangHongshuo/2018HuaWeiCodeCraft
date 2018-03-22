#include "GRU.h"


GRU::GRU()
{

}

GRU::~GRU()
{

}

void GRU::setDims(int hidenDims, int unitNums)
{
    hDim = hidenDims;
    uNum = unitNums;
}

void GRU::setData(vector<vector<double> > &X, vector<vector<double> > &Y)
{
    x = X;
    y = Y;
    xDim = x.size();
    yDim = y.size();
}

void GRU::init()
{
    initCell();
    initCellValue();
}

void GRU::startTrainning()
{
    x = matT(x);
    y = matT(y);
    // Forward
    rValue[0] = matSigmoidF(x[0]*Wr);
    hBarValue[0] = matTanhF(x[0]*W);
    zValue[0] = matSigmoidF(x[0]*Wz);
    hValue[0] = matDotMul(zValue[0],hBarValue[0]);
    yValue[0] = matSigmoidF(hValue[0]*Wy);
    for(int t=1;t<uNum;t++)
    {
        rValue[t] = matSigmoidF((x[t]*Wr)+(hValue[t-1]*Ur));
        hBarValue[t] = matTanhF(((x[t]*W)+(matDotMul(rValue[t],hValue[t-1])))*U);
        zValue[t] = matSigmoidF((x[t]*Wz)+(hValue[t-1]*Uz));
        hValue[t] = matDotMul((1-zValue[t]),hValue[t-1])+matDotMul(zValue[t],hBarValue[t]);
        yValue[t] = matSigmoidF(hValue[t]*Wy);
    }
    // Backward
    for(int t=uNum-1;t>=2;t--)
    {
        delta_y = matDotMul((yValue[t]-y[t]),matSigmoidB(yValue[t]));
        delta_h = delta_y*matT(Wy)+delta_z_Next*matT(Uz)+matDotMul(delta_Next*matT(U),rValue[t+1])+delta_r_Next*matT(Ur)+
                matDotMul(delta_h_Next,(1-zValue[t+1]));
        delta_z = matDotMul(delta_h,(hBarValue[t]-hValue[t-1]),matSigmoidB(zValue[t]));
        delta   = matDotMul(delta_h,zValue[t],matTanhB(hBarValue[t]));
        delta_r = matDotMul(hValue[t-1],(matDotMul(delta_h,zValue[t],matTanhB(hBarValue[t]))*matT(U)),matSigmoidB(rValue[t]));

        dWy = dWy+matT(hValue[t])*delta_y;
        dWz = dWz+matT(x[t])*delta_z;
        dUz = dUz+matT(hValue[t])*delta_z;
        dW  = dW+matT(x[t])*delta;
        dU  = dU+matT(matDotMul(rValue[t],hValue[t-1]))*delta;
        dWr = dWr+matT(x[t])*delta_r;
        dUr = dUr+matT(hValue[t-1])*delta_r;

        delta_r_Next = delta_r;
        delta_z_Next = delta_z;
        delta_h_Next = delta_h;
        delta_Next   = delta;
    }
}

double GRU::sigmoidForward(double x)
{
    return 1.0/(1.0+exp(-x));
}

double GRU::sigmoidBackWard(double x)
{
    return sigmoidForward(x)/(1.0-sigmoidForward(x));
}

vector<vector<double> > GRU::matSigmoidF(const vector<vector<double> > &mat)
{
    vector<vector<double>> output(mat.size());
    for(uint i=0;i<mat.size();i++)
        output[i].resize(mat[0].size());
    for(uint i=0;i<mat.size();i++)
        for(uint j=0;j<mat[0].size();j++)
            output[i][j] = sigmoidForward(mat[i][j]);
    return output;
}

vector<double> GRU::matSigmoidF(const vector<double> &mat)
{
    vector<double> output(mat.size());
    for(uint i=0;i<mat.size();i++)
            output[i] = sigmoidForward(mat[i]);
    return output;
}

vector<vector<double> > GRU::matSigmoidB(const vector<vector<double> > &mat)
{
    vector<vector<double>> output(mat.size());
    for(uint i=0;i<mat.size();i++)
        output[i].resize(mat[0].size());
    for(uint i=0;i<mat.size();i++)
        for(uint j=0;j<mat[0].size();j++)
            output[i][j] = sigmoidBackWard(mat[i][j]);
    return output;
}

vector<double> GRU::matSigmoidB(const vector<double> &mat)
{
    vector<double> output(mat.size());
    for(uint i=0;i<mat.size();i++)
            output[i] = sigmoidBackWard(mat[i]);
    return output;
}

double GRU::tanhForward(double x)
{
    return 2.0/(1.0+exp(-2.0*x))-1.0;
}

double GRU::tanhBackward(double x)
{
    return 1.0-pow(tanhForward(x),2);
}

vector<vector<double> > GRU::matTanhF(const vector<vector<double> > &mat)
{
    vector<vector<double>> output(mat.size());
    for(uint i=0;i<mat.size();i++)
        output[i].resize(mat[0].size());
    for(uint i=0;i<mat.size();i++)
        for(uint j=0;j<mat[0].size();j++)
            output[i][j] = tanhForward(mat[i][j]);
    return output;
}

vector<double> GRU::matTanhF(const vector<double> &mat)
{
    vector<double> output(mat.size());
    for(uint i=0;i<mat.size();i++)
            output[i] = tanhForward(mat[i]);
    return output;
}

vector<vector<double> > GRU::matTanhB(const vector<vector<double> > &mat)
{
    vector<vector<double>> output(mat.size());
    for(uint i=0;i<mat.size();i++)
        output[i].resize(mat[0].size());
    for(uint i=0;i<mat.size();i++)
        for(uint j=0;j<mat[0].size();j++)
            output[i][j] = tanhBackward(mat[i][j]);
    return output;
}

vector<double> GRU::matTanhB(const vector<double> &mat)
{
    vector<double> output(mat.size());
    for(uint i=0;i<mat.size();i++)
            output[i] = tanhBackward(mat[i]);
    return output;
}

// 分配空间
void GRU::initCell()
{
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
        yValue[i].resize(hDim);
    delta_r_Next.resize(hDim);
    delta_z_Next.resize(hDim);
    delta_h_Next.resize(hDim);
    delta_Next.resize(hDim);
}

// 初始化值
void GRU::initCellValue()
{
    for(uint i=0;i<Wy.size();i++)
        for(uint j=0;j<Wy[0].size();j++)
        {
            Wy[i][j] = getRandomValue();
            dWy[i][j] = 0.0;
        }
    for(uint i=0;i<Ur.size();i++)
    {
        for(uint j=0;j<Ur[0].size();j++)
        {
            Ur[i][j] = getRandomValue();
            U[i][j] = getRandomValue();
            Uz[i][j] = getRandomValue();
            dUr[i][j] = 0.0;
            dU[i][j] = 0.0;
            dUz[i][j] = 0.0;
        }
    }
    for(uint i=0;i<Wr.size();i++)
    {
        for(uint j=0;j<Wr[0].size();j++)
        {
            Wr[i][j] = getRandomValue();
            W[i][j] = getRandomValue();
            Wz[i][j] = getRandomValue();
            dWr[i][j] = 0.0;
            dW[i][j] = 0.0;
            dWz[i][j] = 0.0;
        }
    }
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

// 获取[-1,1]随机数
double GRU::getRandomValue()
{
    return (rand()%1000/(double)1002)*2-1;
}

// 矩阵加法，只进行过一次单元测试
vector<vector<double> > operator +(const vector<vector<double> > &mat1, const vector<vector<double> > &mat2)
{
    vector<vector<double> > output;
    if(mat1.size() != mat2.size() || mat1[0].size() != mat2[0].size())
    {
        cout << "Error in matrixAdd!" << endl;
        output.resize(1);
        output[0].resize(1);
        output[0][0] = -1;
        return output;
    }
    else
    {
        output.resize(mat1.size());
        for(uint i=0;i<mat1.size();i++)
            output[i].resize(mat1[0].size());
        for(uint i=0;i<mat1.size();i++)
        {
            for(uint j=0;j<mat1[0].size();j++)
            {
                output[i][j] = mat1[i][j]+mat2[i][j];
            }
        }
        return output;
    }
}

// 矩阵减法，只进行过一次单元测试
vector<vector<double> > operator -(const vector<vector<double> > &mat1, const vector<vector<double> > &mat2)
{
    vector<vector<double> > output;
    if(mat1.size() != mat2.size() || mat1[0].size() != mat2[0].size())
    {
        cout << "Error in matrixSub!" << endl;
        output.resize(1);
        output[0].resize(1);
        output[0][0] = -1;
        return output;
    }
    else
    {
        output.resize(mat1.size());
        for(uint i=0;i<mat1[0].size();i++)
            output[i].resize(mat1[0].size());
        for(uint i=0;i<mat1.size();i++)
        {
            for(uint j=0;j<mat1[0].size();j++)
            {
                output[i][j] = mat1[i][j]-mat2[i][j];
            }
        }
        return output;
    }
}

// 矩阵点除，只进行过一次单元测试
vector<vector<double> > matDotDiv(const vector<vector<double> > &mat1, const vector<vector<double> > &mat2)
{
    vector<vector<double> > output;
    if(mat1.size() != mat2.size() || mat1[0].size() != mat2[0].size())
    {
        cout << "Error in matrixDotDiv!" << endl;
        output.resize(1);
        output[0].resize(1);
        output[0][0] = -1;
        return output;
    }
    else
    {
        output.resize(mat1.size());
        for(uint i=0;i<mat1[0].size();i++)
            output[i].resize(mat1[0].size());
        for(uint i=0;i<mat1.size();i++)
        {
            for(uint j=0;j<mat1[0].size();j++)
            {
                output[i][j] = mat1[i][j]/mat2[i][j];
            }
        }
        return output;
    }
}

// 矩阵乘法，只进行过一次单元测试
vector<vector<double> > operator *(const vector<vector<double> > &mat1, const vector<vector<double> > &mat2)
{
    vector<vector<double> > output;
    if(mat1[0].size() != mat2.size())
    {
        cout << "Error in matrixMul!" << endl;
        output.resize(1);
        output[0].resize(1);
        output[0][0] = -1;
        return output;
    }
    else
    {
        output.resize(mat1.size());
        for(uint i=0;i<output.size();i++)
        {
            output[i].resize(mat2[0].size());
            output[i].assign(output[i].size(),0.0);
        }
        for(uint i=0;i<mat1.size();i++)
        {
            for(uint j=0;j<mat2[0].size();j++)
            {
                for(uint k=0;k<mat1[0].size();k++)
                {
                    output[i][j] += (mat1[i][k]*mat2[k][j]);
                }
            }
        }
        return output;
    }
}

// 矩阵点乘，只进行过一次单元测试
vector<vector<double> > matDotMul(const vector<vector<double> > &mat1, const vector<vector<double> > &mat2)
{
    vector<vector<double> > output;
    if(mat1.size() != mat2.size() || mat1[0].size() != mat2[0].size())
    {
        cout << "Error in matrixDotMul!" << endl;
        output.resize(1);
        output[0].resize(1);
        output[0][0] = -1;
        return output;
    }
    else
    {
        output.resize(mat1.size());
        for(uint i=0;i<mat1[0].size();i++)
            output[i].resize(mat1[0].size());
        for(uint i=0;i<mat1.size();i++)
        {
            for(uint j=0;j<mat1[0].size();j++)
            {
                output[i][j] = mat1[i][j]*mat2[i][j];
            }
        }
        return output;
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

// 常数与矩阵相加，未单元测试
vector<vector<double> > operator +(double a, const vector<vector<double> > &mat2)
{
    vector<vector<double> > output;
    output.resize(mat2.size());
    for(uint i=0;i<mat2[0].size();i++)
        output[i].resize(mat2[0].size());
    for(uint i=0;i<mat2.size();i++)
    {
        for(uint j=0;j<mat2[0].size();j++)
        {
            output[i][j] = a+mat2[i][j];
        }
    }
    return output;
}

// 常数与矩阵相减，未单元测试
vector<vector<double> > operator -(double a, const vector<vector<double> > &mat2)
{
    vector<vector<double> > output;
    output.resize(mat2.size());
    for(uint i=0;i<mat2[0].size();i++)
        output[i].resize(mat2[0].size());
    for(uint i=0;i<mat2.size();i++)
    {
        for(uint j=0;j<mat2[0].size();j++)
        {
            output[i][j] = a-mat2[i][j];
        }
    }
    return output;
}

// 向量和矩阵相乘，只进行过一次单元测试
vector<double> operator *(const vector<double> &mat1, const vector<vector<double> > &mat2)
{
    vector<double> output;
    if(mat1.size() != mat2.size())
    {
        cout << "Error in 1D & 2D matrixMul!" << endl;
        output.resize(1);
        output[0] = -1;
        return output;
    }
    else
    {
        output.resize(mat2[0].size());
        output.assign(mat2[0].size(),0.0);
        for(uint i=0;i<mat2[0].size();i++)
        {
            for(uint j=0;j<mat1.size();j++)
            {
                output[i] += (mat1[j]*mat2[j][i]);
            }
        }
        return output;
    }
}

// 向量和向量点乘，未单元测试
vector<double> matDotMul(const vector<double> &mat1, const vector<double> &mat2)
{
    vector<double> output;
    if(mat1.size() != mat2.size())
    {
        cout << "Error in 1D & 1D matrixDotMul!" << endl;
        output.resize(1);
        output[0] = -1;
        return output;
    }
    else
    {
        output.resize(mat1.size());
        for(uint i=0;i<mat1.size();i++)
            output[i] = mat1[i]*mat2[i];
        return output;
    }
}

// 常数减向量，未单元测试
vector<double> operator -(double a, const vector<double> &mat2)
{
    vector<double> output;
    output.resize(mat2.size());

    for(uint i=0;i<mat2.size();i++)
    {
        output[i] = a-mat2[i];
    }
    return output;
}

// 向量相加，未单元测试
vector<double> operator +(const vector<double> &mat1, const vector<double> &mat2)
{
    vector<double> output;
    if(mat1.size() != mat2.size())
    {
        cout << "Error in 1D & 1D matrixAdd!" << endl;
        output.resize(1);
        output[0] = -1;
        return output;
    }
    else
    {
        output.resize(mat1.size());
        for(uint i=0;i<mat1.size();i++)
            output[i] = mat1[i]+mat2[i];
        return output;
    }
}

// 向量减法，未单元测试
vector<double> operator -(const vector<double> &mat1, const vector<double> &mat2)
{
    vector<double> output;
    if(mat1.size() != mat2.size())
    {
        cout << "Error in 1D & 1D matrixSub!" << endl;
        output.resize(1);
        output[0] = -1;
        return output;
    }
    else
    {
        output.resize(mat1.size());
        for(uint i=0;i<mat1.size();i++)
            output[i] = mat1[i]-mat2[i];
        return output;
    }
}

// 三向量点乘，未单元测试
vector<double> matDotMul(const vector<double> &mat1, const vector<double> &mat2, const vector<double> &mat3)
{
    vector<double> output;
    if(mat1.size() == mat2.size() && mat1.size() == mat3.size())
    {
        output.resize(mat1.size());
        for(uint i=0;i<mat1.size();i++)
            output[i] = mat1[i]*mat2[i]*mat3[i];
        return output;
    }
    else
    {
        cout << "Error in 1D & 1D & 1D matrixDotMul!" << endl;
        output.resize(1);
        output[0] = -1;
        return output;
    }
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

// 矩阵和向量（或常数）相乘，未单元测试
vector<vector<double> > operator *(const vector<vector<double> > &mat1, const vector<double> &mat2)
{
    vector<vector<double>> output;
    if(mat2.size() == 1)
    {
        output.resize(mat1.size());
        for(uint i=0;i<mat1.size();i++)
            output[i].resize(mat1[0].size());
        for(uint i=0;i<mat1.size();i++)
            for(uint j=0;j<mat1[0].size();j++)
                output[i][j] = mat1[i][j]*mat2[0];
        return output;
    }
    else
    {
        if(mat1[0].size() != 1)
        {
            cout << "Error in 2D & 1D matrixMul!" << endl;
            output.resize(1);
            output[0].resize(1);
            output[0][0] = -1;
            return output;
        }
        else
        {
            output.resize(mat1.size());
            for(uint i=0;i<mat1.size();i++)
                output[i].resize(mat2.size());
            for(uint i=0;i<mat1.size();i++)
                for(uint j=0;j<mat1[0].size();j++)
                    output[i][j] = mat1[i][0]*mat2[j];
            return output;
        }
    }

}
