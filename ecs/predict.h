#ifndef __ROUTE_H__
#define __ROUTE_H__

#include <fstream>
#include <iostream>
#include <vector>
#include <math.h>
#include <string.h>
#include <string>
#include "lib_io.h"

#define CPU 67
#define MEM 77

using namespace std;

struct date
{
    int Y = 0;
    int M = 0;
    int D = 0;
    date() {}
};

struct phyServerInfo
{
    int CPUCount = 0;
    int MEMCount = 0;
    int flavorTypeCount = 0;
    // 为方便，索引从1开始，0为无效数据
    int flavorType[16] = {0};
    int optimizedTarget = 0;
    date predictStartTime;
    date predictEndTime;
    phyServerInfo() {}
};

struct trainData
{
    date time;
    // 为方便，索引从1开始，0为无效数据
    int flavorCount[16] = {0};
    trainData() {}
};

struct phyServer
{
    int usedCPU = 0;
    int usedMEM = 0;
    int flavorCount[16] = {0};
    int VMCount = 0;
    bool isFull = false;
    phyServer() {}
};

void predict_server(char * info[MAX_INFO_NUM], char * data[MAX_DATA_NUM], int data_num, char * filename);
void loadInfo(char * info[MAX_INFO_NUM], phyServerInfo &target);
void sortFlavorOrderByOptimizationTarget(phyServerInfo &target);
void quickSort(int left, int right, int * array);
void quickSort(int left, int right, int * array , int * index);
void loadTrainDataToVector(vector<trainData> &target, int daysCount, char * data[MAX_DATA_NUM], int dataLineCount, phyServerInfo &serverInfo);
char * charToNum(char * str, int &target);
void numToDate(int num, date &target);
char * jumpToNextCharBlock(char * str);
int getTrainDataInterval(char * data[MAX_DATA_NUM], int dataNum);
int operator -(const date &to, const date &from);
int getDaysCountInMonth(int Year, int Month);
int getDaysCountInYear(int Year);
bool isFlavorInPhyServerInfo(phyServerInfo &info, int flavorTpye);
bool operator !=(date &a, date &b);

void twoDModel(double (&predictArray)[16][2], vector<trainData> &vTrainData, int trainDataDayCount, int predictDaysCount, phyServerInfo &serverInfo);
void predictSimpleModel(int (&predictArray)[16][2], int (&trainArray)[16][2], int flavorTypeCount, int trainDataDayCount, int predictDaysCount);
void predictAverageModel(int (&predictArray)[16][2], int (&trainArray)[16][2], int flavorTypeCount, int trainDataDayCount, int predictDaysCount);
void ESModel(double (&predictArray)[16][2], vector<trainData> &vTrainData, int trainDataDayCount, int predictDaysCount, phyServerInfo &serverInfo);

void allocateModel(vector<phyServer> &server, int (&predictArray)[16][2], int &predictVMCount, phyServerInfo &serverInfo, int &predictPhyServerCount);

#endif
