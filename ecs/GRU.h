#ifndef GRU_H
#define GRU_H
#include <iostream>
#include <time.h>
#include <vector>
#include <math.h>
#include <float.h>

using namespace std;

vector<vector<double>> operator +(const vector<vector<double>> &mat1, const vector<vector<double>> &mat2);
vector<double> operator +(const vector<double> &mat1, const vector<double> &mat2);
vector<vector<double>> operator +(double a, const vector<vector<double>> &mat2);
void matAdd(const vector<double> &vec1, const vector<double> &vec2, vector<double> &vecOutput);
void matAdd(const vector<vector<double>> &vecPack,int packSize, vector<double> &vecOutput);
void matAdd(const vector<vector<double>> &mat1,const vector<vector<double>> &mat2, vector<vector<double>> &matOutput);

vector<vector<double>> operator -(const vector<vector<double>> &mat1, const vector<vector<double>> &mat2);
vector<double> operator -(const vector<double> &mat1, const vector<double> &mat2);
vector<vector<double>> operator -(double a, const vector<vector<double>> &mat2);
vector<double> operator -(double a, const vector<double> &mat2);
void matSub(const vector<double> &vec1, const vector<double> &vec2, vector<double> &vecOutput);
void matSub(double a, const vector<double> &vec, vector<double> &vecOutput);

vector<vector<double>> matDotDiv(const vector<vector<double>> &mat1, const vector<vector<double>> &mat2);

vector<vector<double>> operator *(const vector<vector<double>> &mat1, const vector<vector<double>> &mat2);
vector<vector<double> > operator *(const vector<vector<double> > &mat1, const vector<double> &mat2);
vector<double> operator *(const vector<double> &mat1, const vector<vector<double>> &mat2);
vector<double> operator *(const vector<double> &mat1, const vector<double> &mat2);
vector<vector<double>> operator *(double a, const vector<vector<double>> &mat2);
void matMul(const vector<double> &vec, const vector<vector<double>> &mat, vector<double> &vecOutput);
void matMul(const vector<vector<double>> &mat, const vector<double> &vec, vector<vector<double>> &matOutput);

vector<vector<double>> matDotMul(const vector<vector<double>> &mat1, const vector<vector<double>> &mat2);
vector<vector<double>> matDotMul(double a, const vector<vector<double>> &mat2);
vector<double> matDotMul(const vector<double> &mat1, const vector<double> &mat2);
vector<double> matDotMul(const vector<double> &mat1, const vector<double> &mat2, const vector<double> &mat3);
void matDotMul(const vector<double> &vec1, const vector<double> &vec2, vector<double> &vecOutput);
void matDotMul(const vector<vector<double>> &vecPack, int packSize, vector<double> &vecOutput);
void matDotMul(const vector<vector<double>*> vecPack, int packSize, vector<double> &vecOutput);

vector<vector<double>> matT(const vector<vector<double> > &src);
vector<vector<double>> matT(const vector<double> &src);
vector<double> matT(const vector<vector<double> > &src,int type);
void matT(const vector<vector<double>> &mat, vector<vector<double>> &matOutput);
void matT(const vector<double> &vec, vector<vector<double>> &vecOutput);

vector<double> autoFit(const vector<vector<double>> &mat1);

typedef unsigned int uint;

class GRU
{
public:
    GRU();
    ~GRU();
    void setDims(int hidenDims, int trainNums, int predictNums);
    void setData(vector<vector<double>> &X, vector<vector<double>> &Y, double _step, int _iterateNum, double _targetError);
    void initCell();
    void initCellValue();
    void startTrainning();

    double sigmoidForward(double x);
    double sigmoidBackward(double x);
    vector<vector<double>> matSigmoidF(const vector<vector<double> > &mat);
    vector<double> matSigmoidF(const vector<double> &mat);
    vector<vector<double>> matSigmoidB(const vector<vector<double>> &mat);
    vector<double> matSigmoidB(const vector<double> &mat);
    void matSigmoidF(const vector<double> &vec, vector<double> &vecOutput);
    void matSigmoidB(const vector<double> &vec, vector<double> &vecOutput);

    double tanhForward(double x);
    double tanhBackward(double x);
    vector<vector<double>> matTanhF(const vector<vector<double> > &mat);
    vector<double> matTanhF(const vector<double> &mat);
    vector<vector<double>> matTanhB(const vector<vector<double> > &mat);
    vector<double> matTanhB(const vector<double> &mat);
    void matTanhF(const vector<double> &vec, vector<double> &vecOutput);
    void matTanhB(const vector<double> &vec, vector<double> &vecOutput);

    void getPredictArray(vector<double> &output);
    vector<vector<double>> x;
    vector<vector<double>> y;

private:
    double getRandomValue();
    double getError(vector<vector<double>> &fit, vector<vector<double>> &target);
    double squrshTo(vector<vector<double> > &src, double a, double b);
    void clearBackwardTempValues();
    int uNum;
    int xDim;
    int yDim;
    int hDim;
    int pNum;
    int iterateNum = -1;
    double step = 0.0;
    double minError = 0.0;
    double error = 0.0;
    double scaleX = 0.0;
    double scaleY = 0.0;
    double targetError = 0.0;

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

    vector<double> delta_r_Next;
    vector<double> delta_z_Next;
    vector<double> delta_h_Next;
    vector<double> delta_Next;

    vector<vector<double>> dWy;
    vector<vector<double>> dWr;
    vector<vector<double>> dUr;
    vector<vector<double>> dW;
    vector<vector<double>> dU;
    vector<vector<double>> dWz;
    vector<vector<double>> dUz;

    vector<double> delta_y;
    vector<double> delta_h;
    vector<double> delta_z;
    vector<double> delta;
    vector<double> delta_r;

    vector<vector<vector<double>>> store;

    // Temp vector
    vector<vector<double>> vTemp_1xh;
    vector<vector<double>> vTemp_1xy;

    vector<vector<double>> mTemp_yxh_1;
    vector<vector<double>> mTemp_hxh_1;
    vector<vector<double>> mTemp_hxy_1;
    vector<vector<double>> mTemp_xxh_1;

    vector<vector<double>*> ptr_vTemp_1xh;
    vector<vector<double>> vTemp_hx1_1;
    vector<vector<double>> vTemp_hx1_2;
    vector<vector<double>> vTemp_xx1;
};

#endif // GRU_H
