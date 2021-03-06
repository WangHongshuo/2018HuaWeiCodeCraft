﻿#include "allocateModel.h"

void allocateModel(vector<vector<phyServer> > &server, int (&predictArray)[19][2], int &predictVMCount, const DataLoader &ecs, vector<int> &predictPhyServerCount )
{
    int tPredictArray_1[19][2];
    int tPredictArray_2[19][2];
    int tPredictVMCount_1 = predictVMCount;
    int tPredictVMCount_2 = predictVMCount;
    memcpy(tPredictArray_1,predictArray,(19*2)*4);
    memcpy(tPredictArray_2,predictArray,(19*2)*4);
    vector<vector<phyServer> > tServer_1;
    vector<vector<phyServer> > tServer_2;
    vector<int> tPServerCount_1;
    vector<int> tPServerCount_2;
    tServer_1.resize(1+ecs.pFlavorTypeCount);
    tServer_2.resize(1+ecs.pFlavorTypeCount);
    tPServerCount_1.resize(1+ecs.pFlavorTypeCount);
    tPServerCount_1.assign(1+ecs.pFlavorTypeCount,0);
    tPServerCount_2.resize(1+ecs.pFlavorTypeCount);
    tPServerCount_2.assign(1+ecs.pFlavorTypeCount,0);
    for(int i=1;i<=ecs.pFlavorTypeCount;i++)
    {
        tServer_1[i].push_back(phyServer(ecs.pFlavor[i].cpu,ecs.pFlavor[i].mem));
        tServer_2[i].push_back(phyServer(ecs.pFlavor[i].cpu,ecs.pFlavor[i].mem));
    }
    allocateModel_1(tServer_1,tPredictArray_1,tPredictVMCount_1,ecs,tPServerCount_1);
    allocateModel_2(tServer_2,tPredictArray_2,tPredictVMCount_2,ecs,tPServerCount_2);
    int tempCPU = 0, tempMEM = 0, sumCPU = 0, sumMEM = 0 ;
    double usage_1, usage_2;
    for(int i=1;i<=ecs.pFlavorTypeCount;i++)
    {
        for(int j=1;j<=tPServerCount_1[i];j++)
        {
            tempCPU += tServer_1[i][j].usedCPU;
            tempMEM += tServer_1[i][j].usedMEM;
            sumCPU += ecs.pFlavor[i].cpu;
            sumMEM += ecs.pFlavor[i].mem;
        }
    }
    usage_1 = double(tempCPU)/double(sumCPU)+double(tempMEM)/double(sumMEM);
    tempCPU = 0, tempMEM = 0, sumCPU = 0, sumMEM = 0;
    for(int i=1;i<=ecs.pFlavorTypeCount;i++)
    {
        for(int j=1;j<=tPServerCount_2[i];j++)
        {
            tempCPU += tServer_2[i][j].usedCPU;
            tempMEM += tServer_2[i][j].usedMEM;
            sumCPU += ecs.pFlavor[i].cpu;
            sumMEM += ecs.pFlavor[i].mem;
        }
    }
    usage_2 = double(tempCPU)/double(sumCPU)+double(tempMEM)/double(sumMEM);
    if(usage_1 > usage_2)
    {
        server = tServer_1;
        memcpy(predictArray,tPredictArray_1,(19*2)*4);
        predictVMCount = tPredictVMCount_1;
        predictPhyServerCount = tPServerCount_1;

    }
    else
    {
        server = tServer_2;
        memcpy(predictArray,tPredictArray_2,(19*2)*4);
        predictVMCount = tPredictVMCount_2;
        predictPhyServerCount = tPServerCount_2;
    }

}
// 分配模型参数格式：物理服务器vector（方便扩充），预测结果数组，预测后的虚拟机总数，物理服务器信息参数，预测需要服务器数量
//  predictArray[i][j]的i<MAX_FLAVOR_TYPE，server的index从1开始
void allocateModel_1(vector<vector<phyServer>> &pServer, int (&predictArray)[19][2], int &predictVMCount,const DataLoader &ecs, vector<int> &pServerCount)
{
    int MAX_FLAVOR_TYPE = ecs.vFlavorTypeCount;
    int tPredictArray[19][2];
    int tPredictVMCount = predictVMCount;
    memcpy(tPredictArray,predictArray,(19*2)*4);

    double maxUsage = -DBL_MAX, tmpUsage;
    double vCpuCount = 0.0, vMemCount = 0.0;
    double pCpuCount, pMemCount;
    int flavorCount, bestChoiceIndex,bestChioceServer = 0,tryCount = 0;
    bool isRestart = false, isGetBestChioce = false;
    double tempDiff, minDiff = DBL_MAX;
    phyServer optPserver(0,0);
    while(tPredictVMCount)
    {
        for(int s=1;s<=ecs.pFlavorTypeCount;s++)
        {
            phyServer tmpPServer(ecs.pFlavor[s].cpu,ecs.pFlavor[s].mem);
            while(!tmpPServer.isFull && tPredictVMCount > 0)
            {
                minDiff = DBL_MAX;
                for(int i=MAX_FLAVOR_TYPE;i>0;i--)
                {
                    flavorCount = tPredictArray[i][1];
                    tryCount ++;
                    while(flavorCount)
                    {
                        if(tmpPServer.usedCPU+ecs.vFlavor[i].cpu > tmpPServer.MAX_CPU ||
                                tmpPServer.usedMEM+ecs.vFlavor[i].mem > tmpPServer.MAX_MEM)
                        {
                            if(tryCount >= MAX_FLAVOR_TYPE && tPredictVMCount > 0)
                            {
                                tmpPServer.isFull = true;
                                if(tmpPServer.getPercentageOfUsedCpu() > 0.95 ||
                                        tmpPServer.getPercentageOfUsedMem() > 0.95)
                                    tmpPServer.isPerfectlyFull = true;
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
                            tryCount = 1;
                            tmpPServer.usedCPU += ecs.vFlavor[i].cpu;
                            tmpPServer.usedMEM += ecs.vFlavor[i].mem;
                            tempDiff = fabs(tmpPServer.getPercentageOfUsedCpu()-tmpPServer.getPercentageOfUsedMem());
                            if(tempDiff <= minDiff)
                            {
                                minDiff = tempDiff;
                                bestChoiceIndex = i;
                                isGetBestChioce = true;
                            }
                            tmpPServer.usedCPU -= ecs.vFlavor[i].cpu;
                            tmpPServer.usedMEM -= ecs.vFlavor[i].mem;
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
                    if(tmpPServer.usedCPU+ecs.vFlavor[bestChoiceIndex].cpu > tmpPServer.MAX_CPU ||
                            tmpPServer.usedMEM+ecs.vFlavor[bestChoiceIndex].mem > tmpPServer.MAX_MEM)
                    {
                        tryCount = 0;
                        minDiff = DBL_MAX;
                        isGetBestChioce = false;
                        cout << "Get Bad Chioce!" << endl;
                        break;
                    }
                    else
                    {
                        tmpPServer.addFlavor(ecs.vFlavor[bestChoiceIndex]);
                        tPredictArray[bestChoiceIndex][1]--;
                        tPredictVMCount--;
                        tryCount = 0;
                        minDiff = DBL_MAX;
                        isGetBestChioce = false;
                    }
                }
//                cout << "Server[" << bestChioceServer << ", " << tmpPServerCount[bestChioceServer] << "] add Flavor[" << ecs.vFlavor[bestChoiceIndex].type << "]:" << endl;
//                cout << "Flavor[" << ecs.vFlavor[bestChoiceIndex].type << "] count: " << tPredictArray[bestChoiceIndex][1] << endl;
//                cout << "server[" << bestChioceServer << ", " << tmpPServerCount[bestChioceServer] << "] used CPU: " <<
//                        server[bestChioceServer][tmpPServerCount[bestChioceServer]].usedCPU << " used MEM: " << tmpServer[bestChioceServer][tmpPServerCount[bestChioceServer]].usedMEM
//                     << " server is full = " << tmpServer[bestChioceServer][tmpPServerCount[bestChioceServer]].isFull << endl;
//                cout <<  "server[" << bestChioceServer << ", " << tmpPServerCount[bestChioceServer] << "] used CPU: " << tmpServer[bestChioceServer][tmpPServerCount[bestChioceServer]].getPercentageOfUsedCpu()*100 << "%, " <<
//                         "used MEM: " << tmpServer[bestChioceServer][tmpPServerCount[bestChioceServer]].getPercentageOfUsedMem()*100 << "%" << endl;
//                cout << "=================" << endl;
//                system("pause");
            }
            // 计算该方案的得分并保存和初始化变量
            // 计算得分
            pCpuCount = tmpPServer.MAX_CPU;
            pMemCount = tmpPServer.MAX_MEM;
            vCpuCount = tmpPServer.usedCPU;
            vMemCount = tmpPServer.usedMEM;
            tmpUsage = vCpuCount/pCpuCount+vMemCount/pMemCount;
            // 保存最优
            if(tmpUsage > maxUsage)
            {
                maxUsage = tmpUsage;
                bestChioceServer = s;
                optPserver = tmpPServer;
            }
            // 清理
            tryCount = 0;
            isRestart = false;
            isGetBestChioce = false;
            minDiff = DBL_MAX;
            for(int i=1;i<=ecs.vFlavorTypeCount;i++)
            {
                tPredictArray[i][1] += tmpPServer.flavorCount[ecs.vFlavor[i].type];
            }
            tPredictVMCount += tmpPServer.VMCount;
        }
        // 保存最优
        maxUsage = -DBL_MAX;
        for(int i=1;i<=ecs.vFlavorTypeCount;i++)
        {
            tPredictArray[i][1] -= optPserver.flavorCount[ecs.vFlavor[i].type];
        }
        tPredictVMCount -= optPserver.VMCount;
        pServer[bestChioceServer].push_back(optPserver);
        pServerCount[bestChioceServer]++;
    }

//    cout << "Before, the predict data count:  VM count: " << predictVMCount << endl;
//    for(int i=1;i<=ecs.vFlavorTypeCount;i++)
//    {
//        cout << "Flavor" << predictArray[i][0] << "  Count: " << predictArray[i][1];
//        cout << endl;
//    }
//    cout << "=================" << endl;

    int maxFlavorCount;
    for(int i=1;i<=ecs.pFlavorTypeCount;i++)
    {
        maxFlavorCount = 0;
        if(pServerCount[i] > 2)
        {
            for(int j=1;j<=ecs.vFlavorTypeCount;j++)
            {
                if(pServer[i][pServerCount[i]].flavorCount[ecs.vFlavor[j].type] > maxFlavorCount)
                    maxFlavorCount = pServer[i][pServerCount[i]].flavorCount[ecs.vFlavor[j].type];
            }
            if(maxFlavorCount < 2)
            {
                for(int j=1;j<=ecs.vFlavorTypeCount;j++)
                {
                    predictArray[j][1] -= pServer[i][pServerCount[i]].flavorCount[ecs.vFlavor[j].type];
                    predictVMCount -= pServer[i][pServerCount[i]].flavorCount[ecs.vFlavor[j].type];
                }
                pServerCount[i] -- ;
                break;
            }
            while(!pServer[i][pServerCount[i]].isFull)
            {
                minDiff = DBL_MAX;
                isGetBestChioce = false;
                tryCount = 0;
                for(int j=1;j<=ecs.vFlavorTypeCount;j++)
                {
                    tryCount++;
                    if(pServer[i][pServerCount[i]].usedCPU+ecs.vFlavor[j].cpu > pServer[i][0].MAX_CPU ||
                            pServer[i][pServerCount[i]].usedMEM+ecs.vFlavor[j].mem > pServer[i][0].MAX_MEM)
                    {
                        if(tryCount >= MAX_FLAVOR_TYPE)
                        {
                            pServer[i][pServerCount[i]].isFull = true;
                            break;
                        }
                        else
                        {
                            continue;
                        }
                    }
                    else
                    {
                        pServer[i][pServerCount[i]].usedCPU += ecs.vFlavor[j].cpu;
                        pServer[i][pServerCount[i]].usedMEM += ecs.vFlavor[j].mem;
                        tempDiff = fabs(pServer[i][pServerCount[i]].getPercentageOfUsedCpu()-pServer[i][pServerCount[i]].getPercentageOfUsedMem());
                        if(tempDiff <= minDiff)
                        {
                            minDiff = tempDiff;
                            bestChoiceIndex = j;
                            isGetBestChioce = true;
                        }
                        pServer[i][pServerCount[i]].usedCPU -= ecs.vFlavor[j].cpu;
                        pServer[i][pServerCount[i]].usedMEM -= ecs.vFlavor[j].mem;
                    }
                }
                if(isGetBestChioce)
                {
                    pServer[i][pServerCount[i]].addFlavor(ecs.vFlavor[bestChoiceIndex]);
                    predictArray[bestChoiceIndex][1] ++;
                    predictVMCount ++;
                }
            }
        }
    }

//    cout << "After, the predict data count:  VM count: " << predictVMCount << endl;
//    for(int i=1;i<=serverInfo.flavorTypeCount;i++)
//    {
//        cout << "Flavor" << predictArray[i][0] << "  Count: " << predictArray[i][1];
//        cout << endl;
//    }
//    cout << "=================" << endl;

//    predictPhyServerCount = SERVER_COUNT;

    cout << "DONE!" << endl;
}

// 分配模型参数格式：物理服务器vector（方便扩充），预测结果数组，预测后的虚拟机总数，物理服务器信息参数，预测需要服务器数量
//  predictArray[i][j]的i<MAX_FLAVOR_TYPE，server的index从1开始
void allocateModel_2(vector<vector<phyServer>> &pServer, int (&predictArray)[19][2], int &predictVMCount,const DataLoader &ecs, vector<int> &pServerCount)
{
    int MAX_FLAVOR_TYPE = ecs.vFlavorTypeCount;
    int tPredictArray[19][2];
    int tPredictVMCount = predictVMCount;
    memcpy(tPredictArray,predictArray,(19*2)*4);
    if(tPredictVMCount > 0)
    {
        for(int i=1;i<=ecs.pFlavorTypeCount;i++)
        {
            pServer[i].push_back(phyServer(ecs.pFlavor[i].cpu,ecs.pFlavor[i].mem));
            pServerCount[i] ++;
        }
    }
    vector<vector<phyServer>> optServer(1+ecs.pFlavorTypeCount);
    vector<vector<phyServer>> tmpServer(1+ecs.pFlavorTypeCount);
    vector<int> optPServerCount(1+ecs.pFlavorTypeCount);
    vector<int> tmpPServerCount(1+ecs.pFlavorTypeCount);
    vector<bool> isApplyNewServer(1+ecs.pFlavorTypeCount);
    for(int i=1;i<=ecs.pFlavorTypeCount;i++)
    {
        tmpServer[i].push_back(phyServer(ecs.pFlavor[i].cpu,ecs.pFlavor[i].mem));
        tmpServer[i].push_back(phyServer(ecs.pFlavor[i].cpu,ecs.pFlavor[i].mem));
        tmpPServerCount[i] ++;
        isApplyNewServer[i] = true;
    }

    double maxUsage = -DBL_MAX, tmpUsage;
    double vCpuCount = 0.0, vMemCount = 0.0;
    double pCpuCount, pMemCount;
    for(int i=1;i<=ecs.vFlavorTypeCount;i++)
    {
        vCpuCount += double(tPredictArray[i][1])*double(ecs.vFlavor[i].cpu);
        vMemCount += double(tPredictArray[i][1])*double(ecs.vFlavor[i].mem);
    }

    // 所有物理服务器组合
    vector<vector<int>> pFlavorGroup;
    vector<int> list(ecs.pFlavorTypeCount);
    for(int i=0;i<ecs.pFlavorTypeCount;i++)
        list[i] = i+1;
    for(int i=1;i<=ecs.pFlavorTypeCount;i++)
    {
        combination(list,i,pFlavorGroup);
    }
    int flavorCount, pServerType, tmpPserverType, bestChoiceIndex, bestChioceServer = 0, tryCount = 0;
    int unallocatedCPU, unallocatedMEM, unusedCPUCount, unusedMEMCount, pFlavorGroupSize;
    bool isRestart = false, isGetBestChioce = false;
    double tempDiff, minDiff = DBL_MAX;
    for(uint c=0;c<pFlavorGroup.size();c++)
    {
        while(tPredictVMCount)
        {
            minDiff = DBL_MAX;
            pFlavorGroupSize = pFlavorGroup[c].size();
            for(uint s=0;s<pFlavorGroup[c].size();s++)
            {
                pServerType = pFlavorGroup[c][s];
                if(!isApplyNewServer[pServerType])
                    continue;
                for(int i=MAX_FLAVOR_TYPE;i>0;i--)
                {
                    flavorCount = tPredictArray[i][1];
                    tryCount ++;
                    while(flavorCount)
                    {
                        if(tmpServer[pServerType][tmpPServerCount[pServerType]].usedCPU+ecs.vFlavor[i].cpu > tmpServer[pServerType][0].MAX_CPU ||
                           tmpServer[pServerType][tmpPServerCount[pServerType]].usedMEM+ecs.vFlavor[i].mem > tmpServer[pServerType][0].MAX_MEM)
                        {
                            if(tryCount >= MAX_FLAVOR_TYPE && tPredictVMCount > 0)
                            {
                                tmpServer[pServerType][tmpPServerCount[pServerType]].isFull = true;
                                if(tmpServer[pServerType][tmpPServerCount[pServerType]].getPercentageOfUsedCpu() > 0.95 ||
                                   tmpServer[pServerType][tmpPServerCount[pServerType]].getPercentageOfUsedMem() > 0.95)
                                {
                                    tmpServer[pServerType][tmpPServerCount[pServerType]].isPerfectlyFull = true;
                                }
                                tryCount = 1;
                                // 计算未分配的flavor和其余pServer可以提供的资源判断是否新开pServer
                                unallocatedCPU = unallocatedMEM = unusedCPUCount = unusedMEMCount = 0;
                                for(uint j=0;j<pFlavorGroup[c].size();j++)
                                {
                                    tmpPserverType = pFlavorGroup[c][j];
                                    if(pFlavorGroupSize > 1)
                                    {
                                        if(pServerType == tmpPserverType || !isApplyNewServer[tmpPserverType])
                                            continue;
                                    }
                                    unusedCPUCount += tmpServer[tmpPserverType][tmpPServerCount[tmpPserverType]].unusedCPU();
                                    unusedMEMCount += tmpServer[tmpPserverType][tmpPServerCount[tmpPserverType]].unusedMEM();
                                }
                                for(int j=1;j<=ecs.vFlavorTypeCount;j++)
                                {
                                    unallocatedCPU += tPredictArray[j][1]*ecs.vFlavor[j].cpu;
                                    unallocatedMEM += tPredictArray[j][1]*ecs.vFlavor[j].mem;
                                }
                                if(unusedCPUCount >= unallocatedCPU && unusedMEMCount >= unallocatedMEM)
                                {
                                    isApplyNewServer[pServerType] = false;
                                    pFlavorGroupSize --;
                                }
                                else
                                {
                                    tmpServer[pServerType].push_back(phyServer(ecs.pFlavor[pServerType].cpu,ecs.pFlavor[pServerType].mem));
                                    tmpPServerCount[pServerType] ++;
                                }
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
                            tryCount = 1;
                            tmpServer[pServerType][tmpPServerCount[pServerType]].usedCPU += ecs.vFlavor[i].cpu;
                            tmpServer[pServerType][tmpPServerCount[pServerType]].usedMEM += ecs.vFlavor[i].mem;
                            tempDiff = fabs(tmpServer[pServerType][tmpPServerCount[pServerType]].getPercentageOfUsedCpu()-tmpServer[pServerType][tmpPServerCount[pServerType]].getPercentageOfUsedMem());
                            if(tempDiff <= minDiff)
                            {
                                minDiff = tempDiff;
                                bestChoiceIndex = i;
                                bestChioceServer = pServerType;
                                isGetBestChioce = true;
                            }
                            tmpServer[pServerType][tmpPServerCount[pServerType]].usedCPU -= ecs.vFlavor[i].cpu;
                            tmpServer[pServerType][tmpPServerCount[pServerType]].usedMEM -= ecs.vFlavor[i].mem;
                            break;
                        }
                    }
                    if(isRestart)
                    {
                        s --;
                        isRestart = false;
                        break;
                    }
                }
            }
            if(isGetBestChioce)
            {
                if(tmpServer[bestChioceServer][tmpPServerCount[bestChioceServer]].usedCPU + ecs.vFlavor[bestChoiceIndex].cpu > tmpServer[bestChioceServer][0].MAX_CPU ||
                   tmpServer[bestChioceServer][tmpPServerCount[bestChioceServer]].usedMEM + ecs.vFlavor[bestChoiceIndex].cpu > tmpServer[bestChioceServer][0].MAX_MEM )
                {
                    tryCount = 0;
                    minDiff = DBL_MAX;
                    isGetBestChioce = false;
                    cout << "Get Bad Chioce!" << endl;
                    break;
                }
                else
                {
                    tmpServer[bestChioceServer][tmpPServerCount[bestChioceServer]].addFlavor(ecs.vFlavor[bestChoiceIndex]);
                    tPredictArray[bestChoiceIndex][1]--;
                    tPredictVMCount--;
                    tryCount = 0;
                    minDiff = DBL_MAX;
                    isGetBestChioce = false;
                }
//                cout << "Server[" << bestChioceServer << ", " << tmpPServerCount[bestChioceServer] << "] add Flavor[" << ecs.vFlavor[bestChoiceIndex].type << "]:" << endl;
//                cout << "Flavor[" << ecs.vFlavor[bestChoiceIndex].type << "] count: " << tPredictArray[bestChoiceIndex][1] << endl;
//                cout << "server[" << bestChioceServer << ", " << tmpPServerCount[bestChioceServer] << "] used CPU: " <<
//                        server[bestChioceServer][tmpPServerCount[bestChioceServer]].usedCPU << " used MEM: " << tmpServer[bestChioceServer][tmpPServerCount[bestChioceServer]].usedMEM
//                        << " server is full = " << tmpServer[bestChioceServer][tmpPServerCount[bestChioceServer]].isFull << endl;
//                cout <<  "server[" << bestChioceServer << ", " << tmpPServerCount[bestChioceServer] << "] used CPU: " << tmpServer[bestChioceServer][tmpPServerCount[bestChioceServer]].getPercentageOfUsedCpu()*100 << "%, " <<
//                         "used MEM: " << tmpServer[bestChioceServer][tmpPServerCount[bestChioceServer]].getPercentageOfUsedMem()*100 << "%" << endl;
//                cout << "=================" << endl;
//                system("pause");
            }
        }
        // 计算该方案的得分并保存和初始化变量
        // 修正物理机数量
        for(int i=1;i<=ecs.pFlavorTypeCount;i++)
        {
            if(tmpServer[i][tmpPServerCount[i]].VMCount == 0)
            {
                tmpPServerCount[i] -- ;
            }
        }
        // 计算得分
        pCpuCount = pMemCount = 0;
        for(int i=1;i<=ecs.pFlavorTypeCount;i++)
        {
            pCpuCount += tmpPServerCount[i]*ecs.pFlavor[i].cpu;
            pMemCount += tmpPServerCount[i]*ecs.pFlavor[i].mem;
        }
        tmpUsage = vCpuCount/pCpuCount+vMemCount/pMemCount;
        // 保存最优
        if(tmpUsage > maxUsage)
        {
            maxUsage = tmpUsage;
            for(int i=1;i<=ecs.pFlavorTypeCount;i++)
            {
                optPServerCount[i] = tmpPServerCount[i];
                optServer[i] = tmpServer[i];
            }
        }
        // 清理
        tryCount = 0;
        isRestart = false;
        isGetBestChioce = false;
        minDiff = DBL_MAX;
        for(int i=1;i<=ecs.pFlavorTypeCount;i++)
        {
            tmpPServerCount[i] = 1;
            isApplyNewServer[i] = true;
            tmpServer[i].clear();
            tmpServer[i].push_back(phyServer(ecs.pFlavor[i].cpu,ecs.pFlavor[i].mem));
            tmpServer[i].push_back(phyServer(ecs.pFlavor[i].cpu,ecs.pFlavor[i].mem));
        }
        tPredictVMCount = predictVMCount;
        memcpy(tPredictArray,predictArray,(19*2)*4);
    }
    // 将最优存放至输出
    for(int i=1;i<=ecs.pFlavorTypeCount;i++)
    {
        pServerCount[i] = optPServerCount[i];
        pServer[i] = optServer[i];
    }

//    cout << "Before, the predict data count:  VM count: " << predictVMCount << endl;
//    for(int i=1;i<=ecs.vFlavorTypeCount;i++)
//    {
//        cout << "Flavor" << predictArray[i][0] << "  Count: " << predictArray[i][1];
//        cout << endl;
//    }
//    cout << "=================" << endl;

    int maxFlavorCount;
    for(int i=1;i<=ecs.pFlavorTypeCount;i++)
    {
        maxFlavorCount = 0;
        if(pServerCount[i] > 2)
        {
            for(int j=1;j<=ecs.vFlavorTypeCount;j++)
            {
                if(pServer[i][pServerCount[i]].flavorCount[ecs.vFlavor[j].type] > maxFlavorCount)
                    maxFlavorCount = pServer[i][pServerCount[i]].flavorCount[ecs.vFlavor[j].type];
            }
            if(maxFlavorCount < 2)
            {
                for(int j=1;j<=ecs.vFlavorTypeCount;j++)
                {
                    predictArray[j][1] -= pServer[i][pServerCount[i]].flavorCount[ecs.vFlavor[j].type];
                    predictVMCount -= pServer[i][pServerCount[i]].flavorCount[ecs.vFlavor[j].type];
                }
                pServerCount[i] -- ;
                break;
            }
            while(!pServer[i][pServerCount[i]].isFull)
            {
                minDiff = DBL_MAX;
                isGetBestChioce = false;
                tryCount = 0;
                for(int j=1;j<=ecs.vFlavorTypeCount;j++)
                {
                    tryCount++;
                    if(pServer[i][pServerCount[i]].usedCPU+ecs.vFlavor[j].cpu > pServer[i][0].MAX_CPU ||
                       pServer[i][pServerCount[i]].usedMEM+ecs.vFlavor[j].mem > pServer[i][0].MAX_MEM)
                    {
                        if(tryCount >= MAX_FLAVOR_TYPE)
                        {
                            pServer[i][pServerCount[i]].isFull = true;
                            break;
                        }
                        else
                        {
                            continue;
                        }
                    }
                    else
                    {
                        pServer[i][pServerCount[i]].usedCPU += ecs.vFlavor[j].cpu;
                        pServer[i][pServerCount[i]].usedMEM += ecs.vFlavor[j].mem;
                        tempDiff = fabs(pServer[i][pServerCount[i]].getPercentageOfUsedCpu()-pServer[i][pServerCount[i]].getPercentageOfUsedMem());
                        if(tempDiff <= minDiff)
                        {
                            minDiff = tempDiff;
                            bestChoiceIndex = j;
                            isGetBestChioce = true;
                        }
                        pServer[i][pServerCount[i]].usedCPU -= ecs.vFlavor[j].cpu;
                        pServer[i][pServerCount[i]].usedMEM -= ecs.vFlavor[j].mem;
                    }
                }
                if(isGetBestChioce)
                {
                    pServer[i][pServerCount[i]].addFlavor(ecs.vFlavor[bestChoiceIndex]);
                    predictArray[bestChoiceIndex][1] ++;
                    predictVMCount ++;
                }
            }
        }
    }

//    cout << "After, the predict data count:  VM count: " << predictVMCount << endl;
//    for(int i=1;i<=serverInfo.flavorTypeCount;i++)
//    {
//        cout << "Flavor" << predictArray[i][0] << "  Count: " << predictArray[i][1];
//        cout << endl;
//    }
//    cout << "=================" << endl;

//    predictPhyServerCount = SERVER_COUNT;

    cout << "DONE!" << endl;
}

template<typename T>
void combination(vector<T> &src, int pick, vector<vector<T> > &dst)
{
    vector<T> comb;
    toCombine(src,0,pick,dst,comb);
}

int factorial(int start, int end)
{
    int ans = 1;
    for(int i=start;i<=end;i++)
        ans *= i;
    return ans;
}

template<typename T>
void toCombine(vector<T> &src, int start, int pick, vector<vector<T> > &dst, vector<T> &comb)
{
    if(pick == 0)
    {
        dst.push_back(comb);
        return;
    }
    for(int i=start;i<=int(src.size())-pick;++i)
    {
        comb.push_back(src[i]);
        toCombine(src,i+1,pick-1,dst,comb);
        comb.pop_back();
    }
}
