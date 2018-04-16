#include "predict.h"
#include <stdio.h>

phyServerInfo serverInfo;
int DAYS[13] = {-1,31,28,31,30,31,30,31,31,30,31,30,31};
vector<FLAVOR> flavor(16);
int trainDataDayCount = 0;
int trainDataIndex = 1;
vector<trainData> trainDataGroup;
int predictDaysCount = 0;
int trainDataFlavorCount[19][2];
int predictDataFlavorCount[19][2];
int predictVMCount = 0;
vector<int> pServerCount;
vector<vector<phyServer>> server;

//你要完成的功能总入口
void predict_server(char * info[MAX_INFO_NUM], char * data[MAX_DATA_NUM], int data_num, char * filename)
{

    DataLoader ecs;
    ecs.loadInfo(info);
    ecs.loadTrainData(ecs.tData,data,data_num);

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
//    cout << "  Y  " << " M" << " D";
//    for(int i=1;i<=ecs.vFlavorTypeCount;i++)
//        cout << " " << ecs.vFlavor[i].type;
//    cout << endl;
//    for(int i=1;i<=ecs.trainDataDaysCount;i++)
//    {
//        cout << ecs.tData[i].time.Y << " " << ecs.tData[i].time.M << " " << ecs.tData[i].time.D << " ";
//        for(int j=1;j<=ecs.vFlavorTypeCount;j++)
//            cout << ecs.tData[i].flavorCount[ecs.vFlavor[j].type] << " ";
//        cout << endl;
//    }
//    cout << "=================" << endl;

    // ======================================================================

    // 训练输入模型：每个flavor在训练数据时间内的总数
    // trainDataFlavorCount初始化为[19][2]
    // trainDataFlavorCount[i][0]为flavorType，[i][1]为训练数据时间范围内该flavorTpye的总数
    for(int i=1;i<=ecs.vFlavorTypeCount;i++)
    {
        trainDataFlavorCount[i][0] = ecs.vFlavor[i].type;
        trainDataFlavorCount[i][1] = 0;
    }
    for(int i=1;i<=ecs.trainDataDaysCount;i++)
    {
        for(int j=1;j<=ecs.vFlavorTypeCount;j++)
        {
            trainDataFlavorCount[j][1] += ecs.tData[i].flavorCount[ecs.vFlavor[j].type];
        }
    }

    // 输出用例（输出全部可输出数据）：
//    cout << "train data count: " << endl;
//    for(int i=1;i<=ecs.vFlavorTypeCount;i++)
//    {
//        cout << "Flavor" << trainDataFlavorCount[i][0] << "  Count: " << trainDataFlavorCount[i][1];
//        cout << endl;
//    }
//    cout << "=================" << endl;

    // ======================================================================

    // 预测
    // predictDataFlavorCount初始化为[19][2]
    // 结构与trainDataFlavorCount相同
    predictDaysCount = ecs.predictDaysCount;
    for(int i=1;i<=ecs.vFlavorTypeCount;i++)
    {
        predictDataFlavorCount[i][0] = ecs.vFlavor[i].type;
        predictDataFlavorCount[i][1] = 0;
    }

    // 预测模型（只可启用一种模型，不启用的模型注释掉）

    // 复杂预测模型：预测每种flavor数量的数组，训练数据vector，训练数据的天数，预测的天数，物理服务器信息
//    predictComplexModel(predictDataFlavorCount,ecs);
    predictModel(predictDataFlavorCount,ecs);
    // 简单预测模型：预测每种flavor数量的数组，训练数据每个flavor数量数组，flavor种类数量，训练数据天数，预测天数
//    predictSimpleModel(predictDataFlavorCount,trainDataFlavorCount,serverInfo.flavorTypeCount,trainDataDayCount,predictDaysCount);

    // 计算虚拟机总数
    for(int i=1;i<=ecs.vFlavorTypeCount;i++)
        predictVMCount += predictDataFlavorCount[i][1];

    // 输出用例（输出全部可输出数据）：
    cout << "predict data count:  VM count: " << predictVMCount << endl;
    for(int i=1;i<=ecs.vFlavorTypeCount;i++)
    {
        cout << "Flavor" << predictDataFlavorCount[i][0] << "  Count: " << predictDataFlavorCount[i][1];
        cout << endl;
    }
    cout << "=================" << endl;

    // ======================================================================

    // 分配预测后的flavor
    server.resize(1+ecs.pFlavorTypeCount);
    pServerCount.resize(1+ecs.pFlavorTypeCount);
    pServerCount.assign(1+ecs.pFlavorTypeCount,0);
    for(int i=1;i<=ecs.pFlavorTypeCount;i++)
    {
        server[i].push_back(phyServer(ecs.pFlavor[i].cpu,ecs.pFlavor[i].mem));
    }
    allocateModel(server,predictDataFlavorCount,predictVMCount,ecs,pServerCount);
    // 输出用例（输出全部可输出数据）：
//    int tempCPU = 0, tempMEM = 0, sumCPU = 0, sumMEM = 0 ;
//    cout << "predicted phy server count: " << endl;
//    for(int i=1;i<=ecs.pFlavorTypeCount;i++)
//    {
//        cout << ecs.pFlavor[i].name << ": " << pServerCount[i] << endl;
//        for(int j=1;j<=pServerCount[i];j++)
//        {
//            cout << "Server " << j << " : " << "CPU: " << server[i][j].usedCPU << "/" << ecs.pFlavor[i].cpu
//                 << ", MEM: " << server[i][j].usedMEM << "/" << ecs.pFlavor[i].mem  << " IsPerfectlyFull: "
//                 << server[i][j].isPerfectlyFull << endl;
//            tempCPU += server[i][j].usedCPU;
//            tempMEM += server[i][j].usedMEM;
//            sumCPU += ecs.pFlavor[i].cpu;
//            sumMEM += ecs.pFlavor[i].mem;
//            for(int k=1;k<=ecs.vFlavorTypeCount;k++)
//            {
//                cout << "Flavor" << ecs.vFlavor[k].type << " " << server[i][j].flavorCount[ecs.vFlavor[k].type] << endl;
//            }
//            cout << endl;
//        }
//    }
//    cout << "Percentage of CPU Usage: " << double(tempCPU)/double(sumCPU) << endl;
//    cout << "Percentage of MEM Usage: " << double(tempMEM)/double(sumMEM) << endl;
//    cout << "=================" << endl;

    // ======================================================================

    // 整理输出到char
    string strOutput;
    strOutput = std::to_string(predictVMCount) + '\n';
    for(int i=1;i<=ecs.vFlavorTypeCount;i++)
    {
        strOutput += "flavor";
        strOutput += std::to_string(ecs.vFlavor[i].type);
        strOutput += ' ';
        strOutput += std::to_string(predictDataFlavorCount[i][1]);
        strOutput += '\n';
    }
    strOutput += '\n';
    for(int s=1;s<=ecs.pFlavorTypeCount;s++)
    {
        strOutput += ecs.pFlavor[s].name;
        strOutput += ' ';
        strOutput += std::to_string(pServerCount[s]);
        strOutput += '\n';
        for(int i=1;i<=pServerCount[s];i++)
        {
            strOutput += ecs.pFlavor[s].name;
            strOutput += '-';
            strOutput += std::to_string(i);
            for(int j=1;j<=ecs.vFlavorTypeCount;j++)
            {
                strOutput += ' ';
                strOutput += "flavor";
                strOutput += std::to_string(ecs.vFlavor[j].type);
                strOutput += ' ';
                strOutput += std::to_string(server[s][i].flavorCount[ecs.vFlavor[j].type]);
            }
            strOutput += '\n';
        }
        strOutput += '\n';
    }
    // 输出用例（输出全部可输出数据）：
    cout << "Test output: " << endl;
    cout << strOutput;

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
