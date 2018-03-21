#ifndef GRU_H
#define GRU_H
#include <iostream>
#include <vector>
#include <math.h>

using namespace std;

vector<vector<double>> operator +(const vector<vector<double>> &mat1, const vector<vector<double>> &mat2);
vector<vector<double>> operator -(const vector<vector<double>> &mat1, const vector<vector<double>> &mat2);
vector<vector<double>> matDotDiv(const vector<vector<double>> &mat1, const vector<vector<double>> &mat2);
vector<vector<double>> operator *(const vector<vector<double>> &mat1, const vector<vector<double>> &mat2);
vector<vector<double>> matDotMul(const vector<vector<double>> &mat1, const vector<vector<double>> &mat2);
vector<vector<double>> matT(vector<vector<double>> &src);

typedef unsigned int uint;

class GRU
{
public:
    GRU();
    ~GRU();
    double sigmoidForward(double x);
    double sigmoidBackWard(double x);
    double tanhForward(double x);
    double tanhBackward(double x);

private:
    void initCell();
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
