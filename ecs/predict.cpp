#include "predict.h"
#include <stdio.h>

phyServerInfo serverInfo;
int DAYS[13] = {-1,31,28,31,30,31,30,31,31,30,31,30,31};
vector<FLAVOR> flavor(16);
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
    for(int i=1;i<16;i++)
    {
        flavor[i].type = i;
        flavor[i].cpu = int(pow(2,(i-1)/3));
        flavor[i].mem = flavor[i].cpu*(int(pow(2,(((i-1)%3)))));
        flavor[i].delta = double(flavor[i].mem)/double(flavor[i].cpu);
    }
    // ======================================================================
    // 载入数据
    loadInfo(info, serverInfo);
    trainDataDayCount  = getTrainDataInterval(data, data_num)+1;
    sortFlavorOrderByOptimizationTarget(serverInfo);

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

    // 输出用例（输出全部可输出数据），数据排列：年 月 日 flavor X的数量
//    cout << " Y   " << "M " << "D";
//    for(int i=1;i<=serverInfo.flavorTypeCount;i++)
//        cout << " " << serverInfo.flavorType[i];
//    cout << endl;
//    for(int i=1;i<=trainDataDayCount;i++)
//    {
//        cout << trainDataGroup[i].time.Y << " " << trainDataGroup[i].time.M << " " << trainDataGroup[i].time.D << " ";
//        for(int j=1;j<=serverInfo.flavorTypeCount;j++)
//            cout << trainDataGroup[i].flavorCount[serverInfo.flavorType[j]] << " ";
//        cout << endl;
//    }
//    cout << "=================" << endl;

    // ======================================================================

    // 训练输入模型：每个flavor在训练数据时间内的总数
    // trainDataFlavorCount初始化为[16][2]
    // trainDataFlavorCount[i][0]为flavorType，[i][1]为训练数据时间范围内该flavorTpye的总数
    for(int i=1;i<=serverInfo.flavorTypeCount;i++)
    {
        trainDataFlavorCount[i][0] = serverInfo.flavorType[i];
        trainDataFlavorCount[i][1] = 0;
    }
    for(int i=1;i<=trainDataDayCount;i++)
    {
        for(int j=1;j<=serverInfo.flavorTypeCount;j++)
        {
            trainDataFlavorCount[j][1] += trainDataGroup[i].flavorCount[serverInfo.flavorType[j]];
        }
    }

    // 输出用例（输出全部可输出数据）：
    cout << "train data count: " << endl;
    for(int i=1;i<=serverInfo.flavorTypeCount;i++)
    {
        cout << "Flavor" << trainDataFlavorCount[i][0] << "  Count: " << trainDataFlavorCount[i][1];
        cout << endl;
    }
    cout << "=================" << endl;

    // ======================================================================

    // 预测
    // predictDataFlavorCount初始化为[16][2]
    // 结构与trainDataFlavorCount相同
    predictDaysCount = serverInfo.predictEndTime-serverInfo.predictStartTime;
    for(int i=1;i<=serverInfo.flavorTypeCount;i++)
    {
        predictDataFlavorCount[i][0] = serverInfo.flavorType[i];
        predictDataFlavorCount[i][1] = 0;
    }

    // 预测模型（只可启用一种模型，不启用的模型注释掉）

    // 复杂预测模型：预测每种flavor数量的数组，训练数据vector，训练数据的天数，预测的天数，物理服务器信息
    predictComplexModel(predictDataFlavorCount,trainDataGroup,trainDataDayCount,predictDaysCount,serverInfo);

    // 简单预测模型：预测每种flavor数量的数组，训练数据每个flavor数量数组，flavor种类数量，训练数据天数，预测天数
//    predictSimpleModel(predictDataFlavorCount,trainDataFlavorCount,serverInfo.flavorTypeCount,trainDataDayCount,predictDaysCount);

    // 计算虚拟机总数
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

    // ======================================================================

    // 分配预测后的flavor
    server.push_back(phyServer(serverInfo.CPUCount,serverInfo.MEMCount));
    allocateModel(server,predictDataFlavorCount,predictVMCount,serverInfo,predictPhyServerCount);

    // 输出用例（输出全部可输出数据）：
    int temp1 = 0, temp2 = 0;
    cout << "predicted phy server count: " << predictPhyServerCount << endl;
    for(int i=1;i<=predictPhyServerCount;i++)
    {
        cout << "Server " << i << " : " << "CPU: " << server[i].usedCPU << "/" << serverInfo.CPUCount
             << ", MEM: " << server[i].usedMEM << "/" << serverInfo.MEMCount  << " IsPerfectlyFull: "
             << server[i].isPerfectlyFull << endl;
        if(serverInfo.optimizedTarget == CPU)
        {
            temp1 += server[i].usedCPU;
            temp2 += serverInfo.CPUCount;
        }
        else
        {
            temp1 += server[i].usedMEM;
            temp2 += serverInfo.MEMCount;
        }
//        for(int j=1;j<=serverInfo.flavorTypeCount;j++)
//        {
//            cout << "Flavor" << serverInfo.flavorType[j] << " " << server[i].flavorCount[serverInfo.flavorType[j]] << endl;
//        }
//        cout << endl;
    }
    cout << "Percentage of Usage: " << double(temp1)/double(temp2) << endl;
    cout << "=================" << endl;

    // ======================================================================

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

// 根据优化目标对phyServer中支持的flavor进行排序
void sortFlavorOrderByOptimizationTarget(phyServerInfo &target)
{
    if(target.optimizedTarget == CPU)
    {
        // 先根据CPU大小排序，后根据MEM大小对相投CPU之间进行微调
        int cpu[16];
        int left,right;
        left = right = 1;
        for(int i=1;i<=target.flavorTypeCount;i++)
            cpu[i] = flavor[target.flavorType[i]].cpu;
        quickSortMinToMax(1,target.flavorTypeCount,cpu,target.flavorType);
//        for(int i=1;i<=target.flavorTypeCount;i++)
//            cout << target.flavorType[i] << " ";
        while(right <= target.flavorTypeCount)
        {
            for(int i=1;i<4;i++)
            {
                if(flavor[target.flavorType[left]].cpu == flavor[target.flavorType[left+i]].cpu)
                {
                    right++;
                    continue;
                }
                else if(left == right)
                {
                    left = right = right+1;
                    break;
                }
                else
                {
                    quickSortMaxToMin(left,right,target.flavorType);
                    left = right = right+1;
                    break;
                }
            }
        }
//        for(int i=1;i<=target.flavorTypeCount;i++)
//            cout << target.flavorType[i] << " ";
//        cout << endl;
    }
    else
    {
        // 先根据MEM大小排序，后根据CPU大小对相投MEM之间进行微调
        int mem[16];
        int left,right;
        left = right = 1;
        for(int i=1;i<=target.flavorTypeCount;i++)
            mem[i] = flavor[target.flavorType[i]].mem;
        quickSortMinToMax(1,target.flavorTypeCount,mem,target.flavorType);
//        for(int i=1;i<=target.flavorTypeCount;i++)
//            cout << target.flavorType[i] << " ";
//        cout << endl;
        while(right <= target.flavorTypeCount)
        {
            for(int i=1;i<4;i++)
            {
                if(flavor[target.flavorType[left]].mem == flavor[target.flavorType[left+i]].mem)
                {
                    right++;
                    continue;
                }
                else if(left == right)
                {
                    left = right = right+1;
                    break;
                }
                else
                {
                    quickSortMaxToMin(left,right,target.flavorType);
                    left = right = right+1;
                    break;
                }
            }
        }
//        for(int i=1;i<=target.flavorTypeCount;i++)
//            cout << target.flavorType[i] << " ";
//        cout << endl;
    }
//    system("pause");
}

// 快速排序
void quickSortMinToMax(int left, int right, int *array)
{
    int flag = 0;
    int l = left;
    int r = right;
    int length = right - left + 1;
    int temp;
    if(length > 1)
    {
        temp = array[left];
        do
        {
            if(flag == 0)
            {
                if(temp <= array[r])
                {
                    r--;
                    continue;
                }
                else
                {
                    array[l] = array[r];
                    flag = 1;
                }
            }
            else
            {
                if(temp >= array[l])
                {
                    l++;
                    continue;
                }
                else
                {
                    array[r] = array[l];
                    flag = 0;
                }
            }
        }while(l < r);
        array[l] = temp;
        quickSortMinToMax(left,l-1,array);
        quickSortMinToMax(r+1,right,array);
    }
}

// 快速排序，根据array大小排序index,array必须和index有相同大小
void quickSortMinToMax(int left, int right, int *array, int *index)
{
    int flag = 0;
    int l = left;
    int r = right;
    int length = right - left + 1;
    int temp,tempIndex;
    if(length > 1)
    {
        temp = array[left];
        tempIndex = index[left];
        do
        {
            if(flag == 0)
            {
                if(temp <= array[r])
                {
                    r--;
                    continue;
                }
                else
                {
                    array[l] = array[r];
                    index[l] = index[r];
                    flag = 1;
                }
            }
            else
            {
                if(temp >= array[l])
                {
                    l++;
                    continue;
                }
                else
                {
                    array[r] = array[l];
                    index[r] = index[l];
                    flag = 0;
                }
            }
        }while(l < r);
        array[l] = temp;
        index[l] = tempIndex;
        quickSortMinToMax(left,l-1,array,index);
        quickSortMinToMax(r+1,right,array,index);
    }
}

// 快速排序
void quickSortMaxToMin(int left, int right, int *array)
{
    int flag = 0;
    int l = left;
    int r = right;
    int length = right - left + 1;
    int temp;
    if(length > 1)
    {
        temp = array[left];
        do
        {
            if(flag == 0)
            {
                if(temp >= array[r])
                {
                    r--;
                    continue;
                }
                else
                {
                    array[l] = array[r];
                    flag = 1;
                }
            }
            else
            {
                if(temp <= array[l])
                {
                    l++;
                    continue;
                }
                else
                {
                    array[r] = array[l];
                    flag = 0;
                }
            }
        }while(l < r);
        array[l] = temp;
        quickSortMaxToMin(left,l-1,array);
        quickSortMaxToMin(r+1,right,array);
    }
}

// 快速排序，根据array大小排序index,array必须和index有相同大小
void quickSortMaxToMin(int left, int right, int *array, int *index)
{
    int flag = 0;
    int l = left;
    int r = right;
    int length = right - left + 1;
    int temp,tempIndex;
    if(length > 1)
    {
        temp = array[left];
        tempIndex = index[left];
        do
        {
            if(flag == 0)
            {
                if(temp >= array[r])
                {
                    r--;
                    continue;
                }
                else
                {
                    array[l] = array[r];
                    index[l] = index[r];
                    flag = 1;
                }
            }
            else
            {
                if(temp <= array[l])
                {
                    l++;
                    continue;
                }
                else
                {
                    array[r] = array[l];
                    index[r] = index[l];
                    flag = 0;
                }
            }
        }while(l < r);
        array[l] = temp;
        index[l] = tempIndex;
        quickSortMaxToMin(left,l-1,array,index);
        quickSortMaxToMin(r+1,right,array,index);
    }
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

// 简单预测模型：预测每种flavor数量的数组，训练数据每个flavor数量数组，flavor种类数量，训练数据天数，预测天数
void predictSimpleModel(int (&predictArray)[16][2], int (&trainArray)[16][2], int flavorTypeCount, int trainDataDayCount, int predictDaysCount)
{
    // 平均法预测模型
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
void allocateModel(vector<phyServer> &server, int (&predictArray)[16][2], int &predictVMCount, phyServerInfo &serverInfo, int &predictPhyServerCount )
{
    int MAX_FLAVOR_TYPE = serverInfo.flavorTypeCount;
    int MAX_CPU = serverInfo.CPUCount;
    int MAX_MEM = serverInfo.MEMCount;
    int tPredictArray[16][2];
    int tPredictVMCount = predictVMCount;
    memcpy(tPredictArray,predictArray,(16*2)*4);
    if(tPredictVMCount > 0)
        server.push_back(phyServer(MAX_CPU,MAX_MEM));
    int SERVER_COUNT = server.size()-1;
    int flavorType = 0, flavorCount, bestChoiceFlavor, bestChoiceIndex,tryCount = 0;
    bool isRestart = false, isGetBestChioce = false;
    double tempDiff, minDiff = DBL_MAX;
    if(serverInfo.optimizedTarget == CPU)
    {
        while(tPredictVMCount)
        {
            minDiff = DBL_MAX;
            for(int i=MAX_FLAVOR_TYPE;i>0;i--)
            {
                flavorType = tPredictArray[i][0];
                flavorCount = tPredictArray[i][1];
                tryCount ++;
                while(flavorCount)
                {
                    if(server[SERVER_COUNT].usedCPU+flavor[flavorType].cpu > MAX_CPU ||
                            server[SERVER_COUNT].usedMEM+flavor[flavorType].mem > MAX_MEM)
                    {
                        if(tryCount >= MAX_FLAVOR_TYPE && tPredictVMCount > 0)
                        {
                            server[SERVER_COUNT].isFull = true;
                            if(server[SERVER_COUNT].getPercentageOfUsedCpu() > 0.95)
                                server[SERVER_COUNT].isPerfectlyFull = true;
                            tryCount = 1;
                            server.push_back(phyServer(MAX_CPU,MAX_MEM));
                            SERVER_COUNT++;
                            isRestart = true;
                            break;
                        }
                        else
                        {
                            break;
                        }
                    }
                    else
                    {
                        server[SERVER_COUNT].usedCPU += flavor[flavorType].cpu;
                        server[SERVER_COUNT].usedMEM += flavor[flavorType].mem;
                        tempDiff = fabs(server[SERVER_COUNT].getPercentageOfUsedMem()-server[SERVER_COUNT].getPercentageOfUsedCpu());
                        if(tempDiff <= minDiff)
                        {
                            minDiff = tempDiff;
                            bestChoiceFlavor = flavorType;
                            bestChoiceIndex = i;
                            isGetBestChioce = true;
                        }
                        server[SERVER_COUNT].usedCPU -= flavor[flavorType].cpu;
                        server[SERVER_COUNT].usedMEM -= flavor[flavorType].mem;
                        break;
                    }
                }
                if(isRestart)
                {
                    isRestart = false;
                    break;
                }
            }
            if(isGetBestChioce)
            {
                if(server[SERVER_COUNT].usedCPU + flavor[bestChoiceFlavor].cpu > MAX_CPU ||
                   server[SERVER_COUNT].usedMEM + flavor[bestChoiceFlavor].mem > MAX_MEM   )
                {
                    tryCount = 0;
                    minDiff = DBL_MAX;
                    isGetBestChioce = false;
                    cout << "Get Bad Chioce!" << endl;
                    break;
                }
                else
                {
                    server[SERVER_COUNT].addFlavor(flavor[bestChoiceFlavor]);
                    tPredictArray[bestChoiceIndex][1]--;
                    tPredictVMCount--;
                    tryCount = 0;
                    minDiff = DBL_MAX;
                    isGetBestChioce = false;
                }
//                cout << "Server[" << SERVER_COUNT << "] add Flavor[" << bestChoiceFlavor << "]:" << endl;
//                cout << "Flavor[" << bestChoiceFlavor << "] count: " << tPredictArray[bestChoiceIndex][1] << endl;
//                cout << "server[" << SERVER_COUNT << "] used CPU: " << server[SERVER_COUNT].usedCPU << " used MEM: " << server[SERVER_COUNT].usedMEM
//                     << " server is full = " << server[SERVER_COUNT].isFull << endl;
//                cout <<  "server[" << SERVER_COUNT << "] used CPU: " << server[SERVER_COUNT].getPercentageOfUsedCpu()*100 << "%, " <<
//                         "used MEM: " << server[SERVER_COUNT].getPercentageOfUsedMem()*100 << "%" << endl;
//                cout << "=================" << endl;
//                system("pause");
            }
        }
    }
    else
    {
        while(tPredictVMCount)
        {
            minDiff = DBL_MAX;
            for(int i=MAX_FLAVOR_TYPE;i>0;i--)
            {
                flavorType = tPredictArray[i][0];
                flavorCount = tPredictArray[i][1];
                tryCount ++;
                while(flavorCount)
                {
                    if(server[SERVER_COUNT].usedCPU+flavor[flavorType].cpu > MAX_CPU ||
                            server[SERVER_COUNT].usedMEM+flavor[flavorType].mem > MAX_MEM)
                    {
                        if(tryCount >= MAX_FLAVOR_TYPE && tPredictVMCount > 0)
                        {
                            server[SERVER_COUNT].isFull = true;
                            if(server[SERVER_COUNT].getPercentageOfUsedCpu() > 0.95)
                                server[SERVER_COUNT].isPerfectlyFull = true;
                            tryCount = 1;
                            server.push_back(phyServer(MAX_CPU,MAX_MEM));
                            SERVER_COUNT++;
                            isRestart = true;
                            break;
                        }
                        else
                        {
                            break;
                        }
                    }
                    else
                    {
                        server[SERVER_COUNT].usedCPU += flavor[flavorType].cpu;
                        server[SERVER_COUNT].usedMEM += flavor[flavorType].mem;
                        tempDiff = fabs(server[SERVER_COUNT].getPercentageOfUsedCpu()-server[SERVER_COUNT].getPercentageOfUsedMem());
                        if(tempDiff <= minDiff)
                        {
                            minDiff = tempDiff;
                            bestChoiceFlavor = flavorType;
                            bestChoiceIndex = i;
                            isGetBestChioce = true;
                        }
                        server[SERVER_COUNT].usedCPU -= flavor[flavorType].cpu;
                        server[SERVER_COUNT].usedMEM -= flavor[flavorType].mem;
                        break;
                    }
                }
                if(isRestart)
                {
                    isRestart = false;
                    break;
                }
            }
            if(isGetBestChioce)
            {
                if(server[SERVER_COUNT].usedCPU + flavor[bestChoiceFlavor].cpu > MAX_CPU ||
                   server[SERVER_COUNT].usedMEM + flavor[bestChoiceFlavor].mem > MAX_MEM   )
                {
                    tryCount = 0;
                    minDiff = DBL_MAX;
                    isGetBestChioce = false;
                    cout << "Get Bad Chioce!" << endl;
                    break;
                }
                else
                {
                    server[SERVER_COUNT].addFlavor(flavor[bestChoiceFlavor]);
                    tPredictArray[bestChoiceIndex][1]--;
                    tPredictVMCount--;
                    tryCount = 0;
                    minDiff = DBL_MAX;
                    isGetBestChioce = false;
                }
//                cout << "Server[" << SERVER_COUNT << "] add Flavor[" << bestChoiceFlavor << "]:" << endl;
//                cout << "Flavor[" << bestChoiceFlavor << "] count: " << tPredictArray[bestChoiceIndex][1] << endl;
//                cout << "server[" << SERVER_COUNT << "] used CPU: " << server[SERVER_COUNT].usedCPU << " used MEM: " << server[SERVER_COUNT].usedMEM
//                     << " server is full = " << server[SERVER_COUNT].isFull << endl;
//                cout <<  "server[" << SERVER_COUNT << "] used CPU: " << server[SERVER_COUNT].getPercentageOfUsedCpu()*100 << "%, " <<
//                         "used MEM: " << server[SERVER_COUNT].getPercentageOfUsedMem()*100 << "%" << endl;
//                cout << "=================" << endl;
//                system("pause");
            }
        }
    }

    cout << "Before, the predict data count:  VM count: " << predictVMCount << endl;
    for(int i=1;i<=serverInfo.flavorTypeCount;i++)
    {
        cout << "Flavor" << predictArray[i][0] << "  Count: " << predictArray[i][1];
        cout << endl;
    }
    cout << "=================" << endl;

    if(SERVER_COUNT > 1)
    {
        int maxCount = 0;
        int temp;
        for(int i=1;i<=MAX_FLAVOR_TYPE;i++)
        {
            temp = server[SERVER_COUNT].flavorCount[serverInfo.flavorType[i]];
            if(temp > maxCount)
            {
                maxCount = temp;
                flavorType = serverInfo.flavorType[i];
            }
        }
        if(maxCount < 2)
        {
            for(int i=1;i<=MAX_FLAVOR_TYPE;i++)
            {
                predictVMCount -= server[SERVER_COUNT].flavorCount[serverInfo.flavorType[i]];
                predictArray[i][1] -=  server[SERVER_COUNT].flavorCount[serverInfo.flavorType[i]];
            }
            SERVER_COUNT--;
        }
        else
        {
            bool isThisFlavorCanPushIn;
            for(int i=MAX_FLAVOR_TYPE;i>0;i--)
            {
                isThisFlavorCanPushIn = true;
                while(isThisFlavorCanPushIn && !server[SERVER_COUNT].isFull)
                {
                    flavorType = serverInfo.flavorType[i];
                    if(server[SERVER_COUNT].usedCPU+flavor[flavorType].cpu > MAX_CPU ||
                            server[SERVER_COUNT].usedMEM+flavor[flavorType].mem > MAX_MEM)
                    {
                        if(i > 1)
                           isThisFlavorCanPushIn = false;
                        else
                            server[SERVER_COUNT].isFull = true;
                    }
                    else
                    {
                        server[SERVER_COUNT].addFlavor(flavor[flavorType]);
                        predictArray[i][1]++;
                        predictVMCount++;
                    }
                }
            }
        }
    }


    cout << "After, the predict data count:  VM count: " << predictVMCount << endl;
    for(int i=1;i<=serverInfo.flavorTypeCount;i++)
    {
        cout << "Flavor" << predictArray[i][0] << "  Count: " << predictArray[i][1];
        cout << endl;
    }
    cout << "=================" << endl;

    predictPhyServerCount = SERVER_COUNT;
    cout << "DONE!" << endl;
}

// 复杂预测模型：预测每种flavor数量的数组，训练数据vector，训练数据的天数，预测的天数，物理服务器信息
void predictComplexModel(int (&predictArray)[16][2], vector<trainData> &vTrainData, int trainDataDayCount, int predictDaysCount, phyServerInfo &serverInfo)
{
    // 将vector数据放入数组中
    // int[i][0]为flavor类型
    // int[i][1]~int[i][trainDataDayCount]为该flavor类型每个索引日期的数量
    vector<vector<int>> trainDataArray(1+serverInfo.flavorTypeCount);
    for(int i=1;i<=serverInfo.flavorTypeCount;i++)
        trainDataArray[i].resize(1+trainDataDayCount+predictDaysCount);
    for(int i=1;i<=serverInfo.flavorTypeCount;i++)
    {
        trainDataArray[i][0] = serverInfo.flavorType[i];
        for(int j=1;j<=trainDataDayCount;j++)
            trainDataArray[i][j] = vTrainData[j].flavorCount[serverInfo.flavorType[i]];
    }

    // 输出用例（输出全部可输出数据）：
//    for(int i=1;i<=serverInfo.flavorTypeCount;i++)
//    {
//        for(int j=0;j<=trainDataDayCount;j++)
//            cout << trainDataArray[i][j] << " ";
//        cout << endl;
//    }
//    cout << "=================" << endl;

    // 输出数据到文件
//    ofstream output("F:/MATLAB_project/HW/test.txt",ios_base::out);
//    for(int i=1;i<=serverInfo.flavorTypeCount;i++)
//    {
//        for(int j=0;j<=trainDataDayCount;j++)
//            output << trainDataArray[i][j] << " ";
//        output << '\n';
//    }
//    system("pause");

    // 可用参数：
    // 数组形式的trainDataArray，int[i][0]为flavor类型，i取值为1~serverInfo.flavorTypeCount
    // int[i][1]~int[i][trainDataDayCount]为该flavor类型每个索引日期的数量
    // 训练数据天数trainDataDayCount，预测天数predictDaysCount
    // 物理服务器信息serverInfo（flavor种类数量serverInfo.flavorTypeCount）
    //　输出参数：predictArray
    // predictArray[i][0]为flavor的类型，已经初始化
    // predictArray[i][1]为该类型的数量，需要输入，i的取值为1~serverInfo.flavorTypeCount
    // TODO

    // 线性累加
    vector<vector<double>> accArray(1+serverInfo.flavorTypeCount);
    for(int i=1;i<=serverInfo.flavorTypeCount;i++)
    {
        accArray[i].resize(1+trainDataDayCount+predictDaysCount);
        accArray[i][1] = trainDataArray[i][1];
        for(int j=2;j<=trainDataDayCount;j++)
            accArray[i][j] = accArray[i][j-1]+trainDataArray[i][j];
    }
    // 设置滑动窗口参数
    vector<double> window(predictDaysCount);
    double delta = predictDaysCount;

    // 滑动线性预测
    double temp = 0.0;
    double step = 0.0001;
    vector<vector<double>> pArray(1+serverInfo.flavorTypeCount);
    for(int i=1;i<=serverInfo.flavorTypeCount;i++)
    {
        double alpha = 0.7;
        int it = 7000;
        double error = DBL_MAX;
        pArray[i].resize(1+trainDataDayCount+predictDaysCount);
        while(it)
        {
            for(int j=0;j<predictDaysCount;j++)
                window[j] = delta*nD(j,double(predictDaysCount)/4)*alpha;
            // 预测
            temp = 0.0;
            for(int j=predictDaysCount+1;j<=trainDataDayCount;j++)
            {
                for(int k=0;k<predictDaysCount;k++)
                    temp += accArray[i][j-predictDaysCount+k]*window[k];
                pArray[i][j] = temp;
                temp = 0.0;
            }
            // 计算误差
            temp = 0.0;
            for(int j=predictDaysCount+1;j<=trainDataDayCount;j++)
                temp += pow(accArray[i][j]-pArray[i][j],2);
            if(temp < error)
            {
                error = temp;
                alpha -= step;
            }
            else
            {
                alpha += step;
                break;
            }
            it--;
        }
        // 预测
        for(int j=predictDaysCount+1;j<=trainDataDayCount+predictDaysCount;j++)
        {
            for(int k=0;k<predictDaysCount;k++)
                temp += accArray[i][j-predictDaysCount+k]*window[k];
            pArray[i][j] = temp;
            temp = 0.0;
            if(j > trainDataDayCount)
                accArray[i][j] = pArray[i][j];
        }
        predictArray[i][1] = ceil(pArray[i][trainDataDayCount+predictDaysCount]-pArray[i][trainDataDayCount+1]);
    }
}

double nD(double in, double sigma)
{
    return 1/sqrt(2*3.1415926)/sigma*exp(-pow(in,2)/2/pow(sigma,2));
}
