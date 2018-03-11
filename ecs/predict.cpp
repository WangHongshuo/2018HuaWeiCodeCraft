#include "predict.h"
#include <stdio.h>

phyServerInfo serverInfo;
int days[13] = {-1,31,28,31,30,31,30,31,31,30,31,30,31};
int trainDataDayCount;
int trainDataIndex = 1;
vector<trainData> trainDataGroup;

//你要完成的功能总入口
void predict_server(char * info[MAX_INFO_NUM], char * data[MAX_DATA_NUM], int data_num, char * filename)
{
    // 载入数据
    loadInfo(info, serverInfo);
    trainDataDayCount  = getTrainDataInterval(data, data_num)+1;

    // 分配vector空间，为方便，索引从1开始，0为无效数据
    trainDataGroup.resize(trainDataDayCount+1);
    loadTrainData(trainDataGroup,trainDataDayCount,data,data_num,serverInfo);

    // 所有索引从1开始
    // serverInfo.flavorTpyeCount为物理服务器可提供的flavor种类数量
    // serverInfo.flavorType[index]为物理服务器可提供的flavor种类,index范围为1~flavorCount
    // trainDataGroup[index]的范围为1~trainDataDayCount, trainDataDayCount由train数据中最后日期和首个日期得出
    // 在载入数据时，只统计serverInfo.flavorType[index]中存在的flavor类型
    // 查看某个index（日期）的某个flavor使用数量：
    // trainDataGroup[index].flavorCount[serverInfo.flavorType[typeIndex]]
    // 输出用例（输出全部可输出数据）：
    for(int i=1;i<=trainDataDayCount;i++)
    {
        cout << trainDataGroup[i].time.Y << " " << trainDataGroup[i].time.M << " " << trainDataGroup[i].time.D << " ";
        for(int j=1;j<=serverInfo.flavorTypeCount;j++)
            cout << trainDataGroup[i].flavorCount[serverInfo.flavorType[j]] << " ";
        cout << endl;
    }
    cout << "=================" << endl;
    // 或者转换为int数组，int[i][0]为flavor类型
    // int[i][1]~int[i][trainDataDayCount]为该flavor类型每个索引日期的数量
    int trainDataArray[serverInfo.flavorTypeCount][trainDataDayCount+1];
    for(int i=0;i<serverInfo.flavorTypeCount;i++)
    {
        trainDataArray[i][0] = serverInfo.flavorType[i+1];
        for(int j=1;j<=trainDataDayCount;j++)
            trainDataArray[i][j] = trainDataGroup[j].flavorCount[serverInfo.flavorType[i+1]];
    }
    // 输出用例（输出全部可输出数据）：
    for(int i=0;i<serverInfo.flavorTypeCount;i++)
    {
        for(int j=0;j<=trainDataDayCount;j++)
            cout << trainDataArray[i][j] << " ";
        cout << endl;
    }

    // ======================================================================
    // TODO


	// 需要输出的内容
	char * result_file = (char *)"17\n\n0 8 0 20";

	// 直接调用输出文件的方法输出到指定文件中(ps请注意格式的正确性，如果有解，第一行只有一个数据；第二行为空；第三行开始才是具体的数据，数据之间用一个空格分隔开)
	write_result(result_file, filename);
}

// 加载info中的数据
void loadInfo(char * info[MAX_INFO_NUM], phyServerInfo &target)
{
    char * pCharTemp = NULL;
    // CPU数量 MEM数量
    pCharTemp = charToNum(info[0],target.CPUCount);
    pCharTemp = charToNum(pCharTemp,target.MEMCount);
    // flavor的种类数量和种类
    pCharTemp = charToNum(info[2],target.flavorTypeCount);
    for(int i=1;i<=target.flavorTypeCount;i++)
    {
        pCharTemp = charToNum(info[2+i],target.flavorType[i]);
    }
    // 优化目标
    if((*info[4+target.flavorTypeCount]) == CPU)
    {
        target.optimizedTarget = CPU;
    }
    else
    {
        target.optimizedTarget = MEM;
    }
    // 预测的起始和终止日期
    int numTypeDate;
    pCharTemp = charToNum(info[6+target.flavorTypeCount],numTypeDate);
    numToDate(numTypeDate,target.predictStartTime);
    pCharTemp = charToNum(info[7+target.flavorTypeCount],numTypeDate);
    numToDate(numTypeDate,target.predictEndTime);
}

// 只转换数字，其他字符跳过，遇到非字符停止
char * charToNum(char * str, int& target)
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

// 以空格或\t为分界跳到下一个字符块
char * jumpToNextCharBlock(char *str)
{
    while((*str) != 32 && (*str) != 9)
    {
        str++;
    }
    return ++str;
}

// 拆分数字格式日期
void numToDate(int num, date &target)
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

// 计算训练数据天数跨度
int getTrainDataInterval(char * data[MAX_DATA_NUM], int dataNum)
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
    returnInterval = getIntervalBetweenTwoDate(start,end);
    return returnInterval;
}

// 计算两个日期之间天数
int getIntervalBetweenTwoDate(const date &from, const date &to)
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

// 获取该月天数
int getDaysCountInMonth(int Year, int Month)
{
    if(Month == 2)
    {
        if((Year%100 == 0 && Year%400 == 0) || (Year%100 != 0 && Year%4 == 0))
            return 29;
        else
            return 28;
    }
    return days[Month];
}

// 获取该年天数
int getDaysCountInYear(int Year)
{
    if((Year%100 == 0 && Year%400 == 0) || (Year%100 != 0 && Year%4 == 0))
        return 366;
    else
        return 365;
}

// 加载训练数据
void loadTrainData(vector<trainData> &target, int daysCount, char *data[], int dataLineCount, phyServerInfo &serverInfo)
{
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
    while(daysIndex <= daysCount)
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
        if(isFlavorInPhyServerInfo(serverInfo,flavorType))
        {
            pCharTemp = charToNum(pCharTemp,numTpyeDate);
            numToDate(numTpyeDate,tempDate);
            while(target[daysIndex].time != tempDate)
            {
                daysIndex++;
            }
            target[daysIndex].flavorCount[flavorType]++;
        }
        dataLineIndex++;
    }
}

// 判断该flavor是否在物理服务器可提供的flavor列表内
bool isFlavorInPhyServerInfo(phyServerInfo &info, int flavorTpye)
{
    for(int i=1;i<=info.flavorTypeCount;i++)
    {
        if(info.flavorType[i] == flavorTpye)
            return true;
    }
    return false;
}

// 重载 != 判断日期是否不相等
bool operator !=(date &a, date &b)
{
    if(a.D != b.D)
        return true;
    if(a.M != b.M)
        return true;
    if(a.Y != b.Y)
        return true;
    return false;
}
