#ifndef ALLOCATEMODEL_H
#define ALLOCATEMODEL_H
#include <fstream>
#include <iostream>
#include <float.h>
#include <vector>
#include <math.h>
#include <string.h>
#include <string>
#include "DataLoader.h"

class phyServer
{
public:
    phyServer(int _MAX_CPU, int _MAX_MEM)
    {
        MAX_CPU = _MAX_CPU;
        MAX_MEM = _MAX_MEM;
    }
    ~phyServer() {}
    int usedCPU = 0;
    int usedMEM = 0;
    int MAX_CPU = 0;
    int MAX_MEM = 0;
    int flavorCount[19] = {0};
    int VMCount = 0;
    bool isFull = false;
    bool isPerfectlyFull = false;
    int unusedCPU()
    {
        return MAX_CPU-usedCPU;
    }
    int unusedMEM()
    {
        return MAX_MEM-usedMEM;
    }
    void loadInfo(int _MAX_CPU, int _MAX_MEM)
    {
        MAX_CPU = _MAX_CPU;
        MAX_MEM = _MAX_MEM;
    }
    double getPercentageOfUsedCpu()
    {
        return double(usedCPU)/double(MAX_CPU);
    }
    double getPercentageOfUsedMem()
    {
        return double(usedMEM)/double(MAX_MEM);
    }
    bool addFlavor(vmFlavor f)
    {
        if(usedCPU+f.cpu > MAX_CPU || usedMEM+f.mem > MAX_MEM)
        {
            return false;
        }
        else
        {
            usedCPU += f.cpu;
            usedMEM += f.mem;
            flavorCount[f.type] ++;
            VMCount ++;
            return true;
        }
    }
    bool removeFlavor(vmFlavor f)
    {
        if(usedCPU-f.cpu < 0 || usedMEM-f.mem < 0)
        {
            return false;
        }
        else
        {
            usedCPU -= f.cpu;
            usedMEM -= f.mem;
            flavorCount[f.type] --;
            VMCount --;
            return true;
        }
    }
private:
};

void allocateModel(vector<vector<phyServer> > &server, int (&predictArray)[19][2], int &predictVMCount, const DataLoader &ecs, vector<int> &predictPhyServerCount );
void allocateModel_1(vector<vector<phyServer> > &server, int (&predictArray)[19][2], int &predictVMCount, const DataLoader &ecs, vector<int> &predictPhyServerCount );
void allocateModel_2(vector<vector<phyServer> > &server, int (&predictArray)[19][2], int &predictVMCount, const DataLoader &ecs, vector<int> &predictPhyServerCount );
template<typename T>
void combination(vector<T> &src, int pick, vector<vector<T>> &dst);
int factorial(int start, int end);
template<typename T>
void toCombine(vector<T> &src, int pick, int count, vector<vector<T>> &dst, vector<T> &comb);

#endif
