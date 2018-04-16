#include "DataLoader.h"



DataLoader::DataLoader()
{
    vFLAVOR.resize(19);
    for(int i=1;i<=18;i++)
    {
        vFLAVOR[i].type = i;
        vFLAVOR[i].cpu = int(pow(2,(i-1)/3));
        vFLAVOR[i].mem = vFLAVOR[i].cpu*(int(pow(2,(((i-1)%3)))));
        vFLAVOR[i].delta = double(vFLAVOR[i].mem)/double(vFLAVOR[i].cpu);
    }
}

DataLoader::~DataLoader()
{

}

void DataLoader::loadInfo(char *info[MAX_INFO_NUM])
{
    char *pCharTemp = NULL;
    int location;
    // 读取物理服务器种类数量
    pCharTemp = charToNum(info[0],pFlavorTypeCount);
    pFlavor.resize(1+pFlavorTypeCount);
    // CPU数量 MEM数量
    for(int i=1;i<=pFlavorTypeCount;i++)
    {
        pCharTemp = info[i];
        pFlavor[i].name = pCharTemp;
        location = pFlavor[i].name.find(" ");
        pFlavor[i].name.resize(location);
        pCharTemp = jumpToNextCharBlock(pCharTemp);
        pCharTemp = charToNum(pCharTemp,pFlavor[i].cpu);
        pCharTemp = charToNum(pCharTemp,pFlavor[i].mem);
    }
    // flavor的种类数量和种类
    pCharTemp = charToNum(info[pFlavorTypeCount+2],vFlavorTypeCount);
    vFlavor.resize(1+vFlavorTypeCount);
    int temp;
    for(int i=1;i<=vFlavorTypeCount;i++)
    {
        pCharTemp = charToNum(info[pFlavorTypeCount+2+i],temp);
        vFlavor[i] = vFLAVOR[temp];
    }
    // 预测的起始和终止日期
    int numTypeDate;
    pCharTemp = charToNum(info[4+pFlavorTypeCount+vFlavorTypeCount],numTypeDate);
    numToDate(numTypeDate,predictStartTime);
    pCharTemp = charToNum(info[5+pFlavorTypeCount+vFlavorTypeCount],numTypeDate);
    numToDate(numTypeDate,predictEndTime);
    predictDaysCount = dateSub(predictEndTime,predictStartTime);
}

void DataLoader::loadTrainData(vector<trainData> &target, char *data[MAX_DATA_NUM], int dataLineCount)
{
    trainDataDaysCount = getTrainDataInterval(data,dataLineCount)+1;
    target.resize(1+trainDataDaysCount);
    int daysIndex = 1, dataLineIndex = 0;
    int numTpyeDate, daysCountInMonth, flavorType;
    char * pCharTemp = NULL;
    // 初始化日期
    pCharTemp = jumpToNextCharBlock(data[0]);
    pCharTemp = jumpToNextCharBlock(pCharTemp);
    pCharTemp = charToNum(pCharTemp,numTpyeDate);
    numToDate(numTpyeDate,target[daysIndex].time);
    daysCountInMonth = getDaysCountInMonth(target[daysIndex].time.Y,target[daysIndex].time.M);
    daysIndex ++;
    while(daysIndex <= trainDataDaysCount)
    {
        target[daysIndex].time.D = target[daysIndex-1].time.D+1;
        target[daysIndex].time.M = target[daysIndex-1].time.M;
        target[daysIndex].time.Y = target[daysIndex-1].time.Y;
        if(target[daysIndex].time.D > daysCountInMonth)
        {
            target[daysIndex].time.D = 1;
            target[daysIndex].time.M += 1;
            if(target[daysIndex].time.M > 12)
            {
                target[daysIndex].time.M = 1;
                target[daysIndex].time.Y += 1;
                daysCountInMonth = getDaysCountInMonth(target[daysIndex].time.Y,target[daysIndex].time.M);
            }
            else
            {
                daysCountInMonth = getDaysCountInMonth(target[daysIndex].time.Y,target[daysIndex].time.M);
            }
        }
        daysIndex++;
    }
    // 从文本中读入数据
    date tempDate;
    daysIndex = 1;
    while (dataLineIndex < dataLineCount)
    {
        pCharTemp = jumpToNextCharBlock(data[dataLineIndex]);
        pCharTemp = charToNum(pCharTemp,flavorType);
        if(isFlavorInPhyServerInfo(vFlavor,flavorType))
        {
            pCharTemp = charToNum(pCharTemp,numTpyeDate);
            numToDate(numTpyeDate,tempDate);
            while(isTheSameDate(target[daysIndex].time,tempDate))
            {
                daysIndex++;
            }
            target[daysIndex].flavorCount[flavorType]++;
        }
        dataLineIndex++;
    }

}

char *DataLoader::charToNum(char *str, int &target)
{
    int sum = 0;
    while(((*str) >= 33) && ((*str) <= 126))
    {
        if(((*str) >= 48) && ((*str) <= 57))
        {
            sum = sum * 10 + ((*str) - '0');
        }
        str++;
    }
    target = sum;
    return ++str;
}

int DataLoader::getTrainDataInterval(char *data[MAX_DATA_NUM], int dataNum)
{
    char * pCharTemp = NULL;
    int s, e, returnInterval;
    date start, end;
    pCharTemp = jumpToNextCharBlock(data[0]);
    pCharTemp = jumpToNextCharBlock(pCharTemp);
    pCharTemp = charToNum(pCharTemp,s);
    pCharTemp = jumpToNextCharBlock(data[dataNum-1]);
    pCharTemp = jumpToNextCharBlock(pCharTemp);
    pCharTemp = charToNum(pCharTemp,e);
    numToDate(s,start);
    numToDate(e,end);
    returnInterval = dateSub(end,start);
    return returnInterval;
}

void DataLoader::numToDate(int num, date &target)
{
    if(num > 0)
    {
        target.Y = num/10000;
        num = num%10000;
        target.M = num/100;
        num = num%100;
        target.D = num;
    }
}

char *DataLoader::jumpToNextCharBlock(char *str)
{
    while((*str) != 32 && (*str) != 9)
    {
        str++;
    }
    return ++str;
}

int DataLoader::dateSub(const DataLoader::date &to, const DataLoader::date &from)
{
    int returnInterval = 0;
    date tempFrom = from;
    date tempTo = to;
    while(tempFrom.Y <= to.Y)
    {
        returnInterval += getDaysCountInYear(tempFrom.Y);
        tempFrom.Y++;
    }
    while(tempFrom.M > 1)
    {
        returnInterval -= getDaysCountInMonth(from.Y,tempFrom.M-1);
        tempFrom.M--;
    }
    while(tempTo.M <= 12)
    {
        returnInterval -= getDaysCountInMonth(to.Y,tempTo.M);
        tempTo.M++;
    }
    returnInterval -= from.D;
    returnInterval += to.D;
    return returnInterval;
}

int DataLoader::getDaysCountInYear(int Year)
{
    if((Year%100 == 0 && Year%400 == 0) || (Year%100 != 0 && Year%4 == 0))
        return 366;
    else
        return 365;
}

int DataLoader::getDaysCountInMonth(int Year, int Month)
{
    if(Month == 2)
    {
        if((Year%100 == 0 && Year%400 == 0) || (Year%100 != 0 && Year%4 == 0))
            return 29;
        else
            return 28;
    }
    return DAYS[Month];
}

bool DataLoader::isFlavorInPhyServerInfo(vector<vmFlavor> &info, int flavorTpye)
{
    for(uint i=0;i<info.size();i++)
    {
        if(info[i].type == flavorTpye)
            return true;
    }
    return false;
}

bool DataLoader::isTheSameDate(const DataLoader::date &a, const DataLoader::date &b)
{
    if(a.D != b.D)
        return true;
    if(a.M != b.M)
        return true;
    if(a.Y != b.Y)
        return true;
    return false;
}

