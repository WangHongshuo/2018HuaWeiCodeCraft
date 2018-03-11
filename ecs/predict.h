#ifndef __ROUTE_H__
#define __ROUTE_H__

#include <iostream>
#include <vector>
#include <cstring>
#include <algorithm>
#include <cstdlib>
#include "lib_io.h"

#define CPU 67
#define MEM 77

using namespace std;

struct date
{
    int Y;
    int M;
    int D;
    date() {}
};

struct phyServerInfo
{
    int CPUCount;
    int MEMCount;
    int flavorCount;
    int flavorType[15];
    int optimizedTarget;
    date predictStartTime;
    date predictEndTime;
    phyServerInfo() {}
};

void predict_server(char * info[MAX_INFO_NUM], char * data[MAX_DATA_NUM], int data_num, char * filename);
void loadInfo(char * info[MAX_INFO_NUM], phyServerInfo &target);
char * charToNum(char * str, int& target);
void numToDate(int num, date &target);
char * jumpToNextCharBlock(char * str);


#endif
