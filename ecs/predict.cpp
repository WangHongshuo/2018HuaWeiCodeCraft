#include "predict.h"
#include <stdio.h>

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

    // ======================================================================
    // 所有索引从1开始
    // ecs.vFlavorTpyeCount需要预测的flavor种类数量
    // ecs.vFlavor[i].type为物理服务器可提供的flavor种类,i范围为1~ecs.vFlavorTpyeCount
    // ecs.tData[i].flavorCount[ecs.vFlavor[j].tpye]查看训练数据第i天某个flavor的数量
    // i范围为1~ecs.trainDataDaysCount,j范围为1~ecs.vFlavorTpyeCount

    // 输出用例（输出全部可输出数据），数据排列：年 月 日 flavor X的数量
//    cout << "  Y  " << " M" << " D";
//    for(int i=1;i<=ecs.vFlavorTypeCount;i++)
//        cout << " " << ecs.vFlavor[i].type;
//    cout << endl;
//    for(int i=1;i<=ecs.trainDataDaysCount;i++)
//    {
//        cout << ecs.tData[i].date.Y << " " << ecs.tData[i].date.M << " " << ecs.tData[i].date.D << " ";
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
    for(int i=1;i<=ecs.vFlavorTypeCount;i++)
    {
        predictDataFlavorCount[i][0] = ecs.vFlavor[i].type;
        predictDataFlavorCount[i][1] = 0;
    }

    // 预测模型（只可启用一种模型，不启用的模型注释掉）

    // 复杂预测模型：预测每种flavor数量的数组，训练数据vector，训练数据的天数，预测的天数，物理服务器信息
    predictModel(predictDataFlavorCount,ecs);
    for(int i=1;i<=ecs.vFlavorTypeCount;i++)
    {
        if(predictDataFlavorCount[i][1] < 0)
            predictDataFlavorCount[i][1] = 0;
    }

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
//    cout << "Test output: " << endl;
//    cout << strOutput;

	// 需要输出的内容
    const char * result_file = strOutput.data();

	// 直接调用输出文件的方法输出到指定文件中(ps请注意格式的正确性，如果有解，第一行只有一个数据；第二行为空；第三行开始才是具体的数据，数据之间用一个空格分隔开)
	write_result(result_file, filename);
}
