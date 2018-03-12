#include "predict.h"
#include <stdio.h>

phyServerInfo serverInfo;
int DAYS[13] = {-1,31,28,31,30,31,30,31,31,30,31,30,31};
int FLAVOR[16][2] = {0,0, 1,1, 1,2, 1,4, 2,2, 2,4, 2,8, 4,4,
                     4,8, 4,16, 8,8, 8,16, 8,32, 16,16, 16,32, 16,64};
double FLAVOR_DELTA[16] = {0, 1, 2, 4, 1, 2, 4, 1,
                           2, 4, 1, 2, 4, 1, 2, 4};
int trainDataDayCount = 0;
int trainDataIndex = 1;
vector<trainData> trainDataGroup;
int predictDaysCount = 0;
int trainDataFlavorCount[16][2];
int predictDataFlavorCount[16][2];
int predictVMCount = 0;
int predictPhyServerCount = 0;
vector<phyServer> server;

//你要完成的功能总入口
void predict_server(char * info[MAX_INFO_NUM], char * data[MAX_DATA_NUM], int data_num, char * filename)
{
    // ======================================================================
    // 载入数据
    loadInfo(info, serverInfo);
    trainDataDayCount  = getTrainDataInterval(data, data_num)+1;

    // 分配vector空间，为方便，索引从1开始，0为无效数据
    trainDataGroup.resize(trainDataDayCount+1);
    loadTrainDataToVector(trainDataGroup,trainDataDayCount,data,data_num,serverInfo);

    // 所有索引从1开始
    // serverInfo.flavorTpyeCount为物理服务器可提供的flavor种类数量
    // serverInfo.flavorType[index]为物理服务器可提供的flavor种类,index范围为1~flavorCount
    // trainDataGroup[index]的范围为1~trainDataDayCount, trainDataDayCount由train数据中最后日期和首个日期得出
    // 在载入数据时，只统计serverInfo.flavorType[index]中存在的flavor类型
    // 查看某个index（日期）的某个flavor使用数量：
    // trainDataGroup[index].flavorCount[serverInfo.flavorType[typeIndex]]

    // 输出用例（输出全部可输出数据）：
//    for(int i=1;i<=trainDataDayCount;i++)
//    {
//        cout << trainDataGroup[i].time.Y << " " << trainDataGroup[i].time.M << " " << trainDataGroup[i].time.D << " ";
//        for(int j=1;j<=serverInfo.flavorTypeCount;j++)
//            cout << trainDataGroup[i].flavorCount[serverInfo.flavorType[j]] << " ";
//        cout << endl;
//    }
//    cout << "=================" << endl;

    // 或者转换为int数组，int[i][0]为flavor类型
    // int[i][1]~int[i][trainDataDayCount]为该flavor类型每个索引日期的数量
    int trainDataArray[serverInfo.flavorTypeCount+1][trainDataDayCount+1];
    for(int i=1;i<=serverInfo.flavorTypeCount;i++)
    {
        trainDataArray[i][0] = serverInfo.flavorType[i];
        for(int j=1;j<=trainDataDayCount;j++)
            trainDataArray[i][j] = trainDataGroup[j].flavorCount[serverInfo.flavorType[i]];
    }

    // 输出用例（输出全部可输出数据）：
//    for(int i=1;i<=serverInfo.flavorTypeCount;i++)
//    {
//        for(int j=0;j<=trainDataDayCount;j++)
//            cout << trainDataArray[i][j] << " ";
//        cout << endl;
//    }
//    cout << "=================" << endl;

    // ======================================================================
    // 训练模型
    // trainDataFlavorCount初始化为[16][2]
    // trainDataFlavorCount[i][0]为flavorType，[i][1]为训练数据时间范围内该flavorTpye的总数
    for(int i=1;i<=serverInfo.flavorTypeCount;i++)
    {
        trainDataFlavorCount[i][0] = trainDataArray[i][0];
        trainDataFlavorCount[i][1] = 0;
        for(int j=1;j<trainDataDayCount;j++)
        {
            trainDataFlavorCount[i][1] += trainDataArray[i][j];
        }
    }

    // 输出用例（输出全部可输出数据）：
//    cout << "train data count: " << endl;
//    for(int i=1;i<=serverInfo.flavorTypeCount;i++)
//    {
//        cout << "Flavor" << trainDataFlavorCount[i][0] << "  Count: " << trainDataFlavorCount[i][1];
//        cout << endl;
//    }
//    cout << "=================" << endl;

    // 预测
    predictDaysCount = serverInfo.predictEndTime-serverInfo.predictStartTime;
    // predictDataFlavorCount初始化为[16][2]
    // 结构与trainDataFlavorCount相同
    for(int i=1;i<=serverInfo.flavorTypeCount;i++)
    {
        predictDataFlavorCount[i][0] = trainDataFlavorCount[i][0];
        predictDataFlavorCount[i][1] = 0;
    }
    // 预测模型：预测后的结果数组，需要训练的数组，flavor种类数量，预测天数
    predictModel(predictDataFlavorCount,trainDataFlavorCount,serverInfo.flavorTypeCount,trainDataDayCount,predictDaysCount);
    for(int i=1;i<=serverInfo.flavorTypeCount;i++)
        predictVMCount += predictDataFlavorCount[i][1];

    // 输出用例（输出全部可输出数据）：
//    cout << "predict data count:  VM count: " << predictVMCount << endl;
//    for(int i=1;i<=serverInfo.flavorTypeCount;i++)
//    {
//        cout << "Flavor" << predictDataFlavorCount[i][0] << "  Count: " << predictDataFlavorCount[i][1];
//        cout << endl;
//    }
//    cout << "=================" << endl;

    // 分配预测后的flavor
    server.push_back(phyServer());
    allocateModel(server,predictDataFlavorCount,predictVMCount,serverInfo,predictPhyServerCount);

    // 输出用例（输出全部可输出数据）：
//    cout << "predicted phy server count: " << predictPhyServerCount << endl;
//    for(int i=1;i<=predictPhyServerCount;i++)
//    {
//        cout << "Server " << i << " :" << endl;
//        for(int j=1;j<=serverInfo.flavorTypeCount;j++)
//        {
//            cout << "Flavor" << serverInfo.flavorType[j] << " " << server[i].flavorCount[serverInfo.flavorType[j]] << endl;
//        }
//        cout << endl;
//    }
//    cout << "=================" << endl;

    // 整理输出到char
    string strOutput;
    strOutput = std::to_string(predictVMCount) + '\n';
    for(int i=1;i<=serverInfo.flavorTypeCount;i++)
    {
        strOutput += "flavor";
        strOutput += std::to_string(serverInfo.flavorType[i]);
        strOutput += ' ';
        strOutput += std::to_string(predictDataFlavorCount[i][1]);
        strOutput += '\n';
    }
    strOutput += '\n';
    strOutput += std::to_string(predictPhyServerCount);
    strOutput += '\n';
    for(int i=1;i<=predictPhyServerCount;i++)
    {
        strOutput += std::to_string(i);
        for(int j=1;j<=serverInfo.flavorTypeCount;j++)
        {
            strOutput += ' ';
            strOutput += "flavor";
            strOutput += std::to_string(serverInfo.flavorType[j]);
            strOutput += ' ';
            strOutput += std::to_string(server[i].flavorCount[serverInfo.flavorType[j]]);
        }
        strOutput += '\n';
    }

    // 输出用例（输出全部可输出数据）：
//    cout << "Test output: " << endl;
//    cout << strOutput;

	// 需要输出的内容
    const char * result_file = strOutput.data();

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
    returnInterval = end-start;
    return returnInterval;
}

// 重载 - ，计算两个日期之间天数
int operator -(const date &to, const date &from)
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
    return DAYS[Month];
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
void loadTrainDataToVector(vector<trainData> &target, int daysCount, char *data[], int dataLineCount, phyServerInfo &serverInfo)
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

// 预测模型参数格式：预测后的结果数组，需要训练的数组，flavor种类数量，训练数据天数，预测天数
// predictArray[i][j]和trainArray[i][j]的i<flavorTypeCount
void predictModel(int (&predictArray)[16][2], int (&trainArray)[16][2], int flavorTypeCount, int trainDataDayCount, int predictDaysCount)
{
    // 平均法预测模型，新建模型可按照该函数格式新建函数
    predictAverageModel(predictArray,trainArray,flavorTypeCount,trainDataDayCount,predictDaysCount);
}

// 平均法预测模型
void predictAverageModel(int (&predictArray)[16][2], int (&trainArray)[16][2], int flavorTypeCount, int trainDataDayCount, int predictDaysCount)
{
    double delta = double(predictDaysCount)/double(trainDataDayCount);
    for(int i=1;i<=flavorTypeCount;i++)
    {
        predictArray[i][1] = int(ceil(double(trainArray[i][1])*delta));
    }
}

// 分配模型参数格式：物理服务器vector（方便扩充），预测结果数组，预测后的虚拟机总数，物理服务器信息参数，预测需要服务器数量
//  predictArray[i][j]的i<MAX_FLAVOR_TYPE，server的index从1开始
void allocateModel(vector<phyServer> &server, int (&predictArray)[16][2], int predictVMCount, phyServerInfo &serverInfo, int &predictPhyServerCount )
{
    if(predictVMCount > 0)
        server.push_back(phyServer());
    int SERVER_COUNT = server.size()-1;
    int MAX_FLAVOR_TYPE = serverInfo.flavorTypeCount;
    int MAX_CPU = serverInfo.CPUCount;
    int MAX_MEM = serverInfo.MEMCount;
    int OPT_TARGET = serverInfo.optimizedTarget;
    int flavorType = 0, flavorCount;
    while(predictVMCount)
    {
        if(OPT_TARGET)
        {
            for(int i=1;i<=MAX_FLAVOR_TYPE;i++)
            {
                flavorType = predictArray[i][0];
                flavorCount = predictArray[i][1];
                while(flavorCount)
                {
                    for(int k=1;k<=SERVER_COUNT;k++)
                    {

                        if(server[k].isFull)
                        {
                            continue;
                        }
                        if(server[k].usedCPU+FLAVOR[flavorType][0] > MAX_CPU ||
                           server[k].usedMEM+FLAVOR[flavorType][1] > MAX_MEM)
                        {
                            server[k].isFull = true;
                        }
                        else
                        {
                            server[k].usedCPU += FLAVOR[flavorType][0];
                            server[k].usedMEM += FLAVOR[flavorType][1];
                            server[k].flavorCount[flavorType]++;
                            flavorCount--;
                            predictVMCount--;
                        }
                        if(flavorCount == 0)
                            break;
                        if(k == SERVER_COUNT && flavorCount > 0 && server[k].isFull)
                        {
                            server.push_back(phyServer());
                            SERVER_COUNT++;
                        }
//                        cout << flavorCount << " " << predictVMCount << endl;
//                        cout << k << " " << server[k].usedCPU << " " << server[k].usedMEM << " " << server[k].isFull << endl;
//                        cout << SERVER_COUNT << endl;
//                        system("pause");
                    }
                }
            }

        }
        else
        {

        }
    }
    predictPhyServerCount = SERVER_COUNT;
}
