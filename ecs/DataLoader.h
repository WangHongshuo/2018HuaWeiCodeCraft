#ifndef DATALOADER_H
#define DATALOADER_H
#include <iostream>
#include <float.h>
#include <math.h>
#include <string.h>
#include <string>
#include <vector>

#define MAX_INFO_NUM    50
#define MAX_DATA_NUM    50000
#define MAX_LINE_LEN 55000
typedef unsigned int uint;

using namespace std;

struct vmFlavor
{
    int type = 0;
    int cpu = 0;
    int mem = 0;
    double delta = 0;
    vmFlavor() {}
};

class DataLoader
{
    struct phyFlavor
    {
        string name;
        int cpu;
        int mem;
        phyFlavor() {}
    };

    struct date
    {
        int Y = 0;
        int M = 0;
        int D = 0;
        date() {}
    };

    struct trainData
    {
        date time;
        // 为方便，索引从1开始，0为无效数据
        int flavorCount[19] = {0};
        trainData() {}
    };



public:
    typedef vector<trainData> trainArray;
    DataLoader();
    ~DataLoader();
    vector<phyFlavor> pFlavor;
    vector<vmFlavor> vFlavor;
    vector<trainData> tData;
    int pFlavorTypeCount = 0;
    int vFlavorTypeCount = 0;
    int predictDaysCount = 0;
    date predictStartTime;
    date predictEndTime;
    int trainDataDaysCount = 0;
    void loadInfo(char *info[MAX_INFO_NUM]);
    void loadTrainData(vector<trainData> &target, char *data[MAX_DATA_NUM], int dataLineCount);
    char *charToNum(char *str, int &target);
    int getTrainDataInterval(char * data[MAX_DATA_NUM], int dataNum);
    void numToDate(int num, date &target);
    char *jumpToNextCharBlock(char *str);
    int dateSub(const date &to, const date &from);
    int getDaysCountInYear(int Year);
    int getDaysCountInMonth(int Year, int Month);
    bool isFlavorInPhyServerInfo(vector<vmFlavor> &info, int flavorTpye);
    bool isTheSameDate(const date &a, const date &b);


private:
    int DAYS[13] = {-1,31,28,31,30,31,30,31,31,30,31,30,31};
    vector<vmFlavor> vFLAVOR;

};

#endif
