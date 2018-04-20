#include "allocateModel.h"

// 分配模型参数格式：物理服务器vector（方便扩充），预测结果数组，预测后的虚拟机总数，物理服务器信息参数，预测需要服务器数量
//  predictArray[i][j]的i<MAX_FLAVOR_TYPE，server的index从1开始
void allocateModel(vector<vector<phyServer>> &pServer, int (&predictArray)[19][2], int &predictVMCount,const DataLoader &ecs, vector<int> &pServerCount)
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
