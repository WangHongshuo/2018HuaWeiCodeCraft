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

double GRU::sigmoidForward(double x)
{
    return 1.0/(1.0+exp(-x));
}

double GRU::sigmoidBackWard(double x)
{
    return sigmoidForward(x)/(1.0-sigmoidForward(x));
}

double GRU::tanhForward(double x)
{
    return 2.0/(1.0+exp(-2.0*x))-1.0;
}

double GRU::tanhBackward(double x)
{
    return 1.0-pow(tanhForward(x),2);
}

// 分配空间
void GRU::initCell()
{
    Wy.resize(hDim);
    for(int i=0;i<hDim;i++)
        Wy[i].resize(yDim);
    Wr.resize(xDim);
    for(int i=0;i<xDim;i++)
        Wr[i].resize(hDim);
    Ur.resize(hDim);
    for(int i=0;i<hDim;i++)
        Ur[i].resize(hDim);
    W.resize(xDim);
    for(int i=0;i<xDim;i++)
        W[i].resize(hDim);
    U.resize(hDim);
    for(int i=0;i<hDim;i++)
        U[i].resize(hDim);
    Wz.resize(xDim);
    for(int i=0;i<xDim;i++)
        Wz[i].resize(hDim);
    Uz.resize(hDim);
    for(int i=0;i<hDim;i++)
        Uz[i].resize(hDim);

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
}

// 初始化值
void GRU::initCellValue()
{
    for(uint i=0;i<Wy.size();i++)
        for(uint j=0;j<Wy[0].size();j++)
            Wy[i][j] = getRandomValue();
    for(uint i=0;i<Ur.size();i++)
    {
        for(uint j=0;j<Ur[0].size();j++)
        {
            Ur[i][j] = getRandomValue();
            U[i][j] = getRandomValue();
            Uz[i][j] = getRandomValue();
        }
    }
    for(uint i=0;i<Wr.size();i++)
    {
        for(uint j=0;j<Wr[0].size();j++)
        {
            Wr[i][j] = getRandomValue();
            W[i][j] = getRandomValue();
            Wz[i][j] = getRandomValue();
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
        for(uint i=0;i<mat1[0].size();i++)
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
vector<vector<double> > matT(vector<vector<double> > &src)
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

