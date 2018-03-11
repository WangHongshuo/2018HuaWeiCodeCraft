#include "predict.h"
#include <stdio.h>

phyServerInfo serverInfo;


//你要完成的功能总入口
void predict_server(char * info[MAX_INFO_NUM], char * data[MAX_DATA_NUM], int data_num, char * filename)
{
    loadInfo(info, serverInfo);

	// 需要输出的内容
	char * result_file = (char *)"17\n\n0 8 0 20";

	// 直接调用输出文件的方法输出到指定文件中(ps请注意格式的正确性，如果有解，第一行只有一个数据；第二行为空；第三行开始才是具体的数据，数据之间用一个空格分隔开)
	write_result(result_file, filename);
}

// 加载info中的数据
void loadInfo(char *info[MAX_INFO_NUM], phyServerInfo &target)
{
    char * pCharTemp = NULL;
    // CPU数量 MEM数量
    pCharTemp = charToNum(info[0],target.CPUCount);
    pCharTemp = charToNum(pCharTemp,target.MEMCount);
//    cout << target.CPUCount << " " << target.MEMCount;
    // flavor的种类数量和种类
    pCharTemp = charToNum(info[2],target.flavorCount);
//    cout << target.flavorCount;
    for(int i=0;i<target.flavorCount;i++)
    {
        pCharTemp = charToNum(info[3+i],target.flavorType[i]);
    }
//    cout << target.flavorType[4];
    // 优化目标
    if((*info[4+target.flavorCount]) == CPU)
    {
        target.optimizedTarget = CPU;
    }
    else
    {
        target.optimizedTarget = MEM;
    }
//    cout << target.optimizedTarget;
    // 预测的起始和终止日期
    int numTypeDate;
    pCharTemp = charToNum(info[6+target.flavorCount],numTypeDate);
    numToDate(numTypeDate,target.predictStartTime);
    pCharTemp = charToNum(info[7+target.flavorCount],numTypeDate);
    numToDate(numTypeDate,target.predictEndTime);
//    cout << target.predictEndTime.Y << " "
//         << target.predictEndTime.M << " "
//         << target.predictEndTime.D;
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
    while((*str) != 32 || (*str) != 9)
    {
        str++;
    }
    return ++str;
}


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
