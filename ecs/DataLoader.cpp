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
    int numTypeDateTime;
    pCharTemp = charToNum(info[4+pFlavorTypeCount+vFlavorTypeCount],numTypeDateTime);
    numToDate(numTypeDateTime,predictBeginDate);
    pCharTemp = charToNum(pCharTemp,numTypeDateTime);
    numToTime(numTypeDateTime,predictBeginTime);
    pCharTemp = charToNum(info[5+pFlavorTypeCount+vFlavorTypeCount],numTypeDateTime);
    numToDate(numTypeDateTime,predictEndDate);
    pCharTemp = charToNum(pCharTemp,numTypeDateTime);
    numToTime(numTypeDateTime,predictEndTime);
    predictDaysCount = dateSub(predictEndDate,predictBeginDate);
    int seconds = timeSub(predictEndTime,predictBeginTime);
    if(double(seconds)/double(24*60*60) > 0.5)
        predictDaysCount += 1;

}

void DataLoader::loadTrainData(vector<trainData> &target, char *data[MAX_DATA_NUM], int dataLineCount)
{
    trainDataDaysCount = getTrainDataInterval(data,dataLineCount)+1;
    target.resize(1+trainDataDaysCount);
    int daysIndex = 1, dataLineIndex = 0;
    int numTpyeDate, daysCountInMonth, flavorType;
    Date tempDate;
    char * pCharTemp = NULL;
    // 获取训练数据最后第一条的时间
    pCharTemp = jumpToNextCharBlock(data[0]);
    pCharTemp = jumpToNextCharBlock(pCharTemp);
    pCharTemp = charToNum(pCharTemp,numTpyeDate);
    numToDate(numTpyeDate,trainBeginDate);
    pCharTemp = charToNum(pCharTemp,numTpyeDate);
    numToTime(numTpyeDate,trainBeginTime);
    trainBeginIndex = 1;
    trainEndIndex = trainDataDaysCount;
    predictBeginIndex = dateSub(predictBeginDate,trainBeginDate)+1;
    if(trainBeginTime.hour > 12)
        predictBeginIndex -= 1;
    predictEndIndex = predictBeginIndex+predictDaysCount-1;

    // 初始化日期
    pCharTemp = jumpToNextCharBlock(data[0]);
    pCharTemp = jumpToNextCharBlock(pCharTemp);
    pCharTemp = charToNum(pCharTemp,numTpyeDate);
    numToDate(numTpyeDate,target[daysIndex].date);
    daysCountInMonth = getDaysCountInMonth(target[daysIndex].date.Y,target[daysIndex].date.M);
    daysIndex ++;
    while(daysIndex <= trainDataDaysCount)
    {
        target[daysIndex].date.D = target[daysIndex-1].date.D+1;
        target[daysIndex].date.M = target[daysIndex-1].date.M;
        target[daysIndex].date.Y = target[daysIndex-1].date.Y;
        if(target[daysIndex].date.D > daysCountInMonth)
        {
            target[daysIndex].date.D = 1;
            target[daysIndex].date.M += 1;
            if(target[daysIndex].date.M > 12)
            {
                target[daysIndex].date.M = 1;
                target[daysIndex].date.Y += 1;
                daysCountInMonth = getDaysCountInMonth(target[daysIndex].date.Y,target[daysIndex].date.M);
            }
            else
            {
                daysCountInMonth = getDaysCountInMonth(target[daysIndex].date.Y,target[daysIndex].date.M);
            }
        }
        daysIndex++;
    }
    // 从文本中读入数据
    daysIndex = 1;
    while (dataLineIndex < dataLineCount)
    {
        pCharTemp = jumpToNextCharBlock(data[dataLineIndex]);
        pCharTemp = charToNum(pCharTemp,flavorType);
        if(isFlavorInPhyServerInfo(vFlavor,flavorType))
        {
            pCharTemp = charToNum(pCharTemp,numTpyeDate);
            numToDate(numTpyeDate,tempDate);
            while(isTheSameDate(target[daysIndex].date,tempDate))
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
    Date start, end;
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

void DataLoader::numToDate(int num, Date &target)
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

void DataLoader::numToTime(int num, DataLoader::Time &target)
{
    if(num > 0)
    {
        target.hour = num/10000;
        num = num%10000;
        target.min = num/100;
        num = num%100;
        target.sec = num;
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

int DataLoader::dateSub(const DataLoader::Date &to, const DataLoader::Date &from)
{
    int returnInterval = 0;
    Date tempFrom = from;
    Date tempTo = to;
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

void DataLoader::dateCopy(const DataLoader::Date &src, DataLoader::Date &dst)
{
    dst.Y = src.Y;
    dst.M = src.M;
    dst.D = src.D;
}

int DataLoader::timeSub(const DataLoader::Time &to, const DataLoader::Time &from)
{
    int returnInterval;
    int h, m, s;
    h = to.hour-from.hour;
    returnInterval = h*60*60;
    m = to.min-from.min;
    returnInterval += m*60;
    s = to.sec-from.sec;
    returnInterval += s;
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

bool DataLoader::isTheSameDate(const DataLoader::Date &a, const DataLoader::Date &b)
{
    if(a.D != b.D)
        return true;
    if(a.M != b.M)
        return true;
    if(a.Y != b.Y)
        return true;
    return false;
}

