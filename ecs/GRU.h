#ifndef GRU_H
#define GRU_H
#include <iostream>
#include <time.h>
#include <vector>
#include <math.h>

using namespace std;

vector<vector<double>> operator +(const vector<vector<double>> &mat1, const vector<vector<double>> &mat2);
vector<vector<double>> operator +(double a, const vector<vector<double>> &mat2);
vector<vector<double>> operator -(const vector<vector<double>> &mat1, const vector<vector<double>> &mat2);
vector<vector<double>> operator -(double a, const vector<vector<double>> &mat2);
vector<vector<double>> matDotDiv(const vector<vector<double>> &mat1, const vector<vector<double>> &mat2);
vector<vector<double>> operator *(const vector<vector<double>> &mat1, const vector<vector<double>> &mat2);
vector<double> operator *(const vector<double> &mat1, const vector<vector<double>> &mat2);
vector<vector<double>> matDotMul(const vector<vector<double>> &mat1, const vector<vector<double>> &mat2);
vector<vector<double>> matT(const vector<vector<double> > &src);

typedef unsigned int uint;

class GRU
{
public:
    GRU();
    ~GRU();
    void setDims(int hidenDims, int unitNums);
    void setData(vector<vector<double>> &X,vector<vector<double>> &Y);
    void init();
    void startTrainning();

    double sigmoidForward(double x);
    double sigmoidBackWard(double x);
    vector<vector<double>> matSigmoidF(const vector<vector<double> > &mat);
    vector<double> matSigmoidF(const vector<double> &mat);
    vector<vector<double>> matSigmoidB(const vector<vector<double>> &mat);
    vector<double> matSigmoidB(const vector<double> &mat);
    double tanhForward(double x);
    double tanhBackward(double x);
    vector<vector<double>> matTanhF(const vector<vector<double> > &mat);
    vector<double> matTanhF(const vector<double> &mat);
    vector<vector<double>> matTanhB(const vector<vector<double> > &mat);
    vector<double> matTanhB(const vector<double> &mat);

    vector<vector<double>> x;
    vector<vector<double>> y;

private:
    void initCell();
    void initCellValue();
    double getRandomValue();
    int uNum;
    int xDim;
    int yDim;
    int hDim;

    vector<vector<double>> Wy;
    vector<vector<double>> Wr;
    vector<vector<double>> Ur;
    vector<vector<double>> W;
    vector<vector<double>> U;
    vector<vector<double>> Wz;
    vector<vector<double>> Uz;

    vector<vector<double>> rValue;
    vector<vector<double>> zValue;
    vector<vector<double>> hBarValue;
    vector<vector<double>> hValue;
    vector<vector<double>> yValue;


};

#endif // GRU_H
