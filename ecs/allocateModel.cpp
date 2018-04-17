#include "allocateModel.h"

// 分配模型参数格式：物理服务器vector（方便扩充），预测结果数组，预测后的虚拟机总数，物理服务器信息参数，预测需要服务器数量
//  predictArray[i][j]的i<MAX_FLAVOR_TYPE，server的index从1开始
void allocateModel(vector<vector<phyServer>> &server, int (&predictArray)[19][2], int &predictVMCount,const DataLoader &ecs, vector<int> &pServerCount)
{
    int MAX_FLAVOR_TYPE = ecs.vFlavorTypeCount;
    int tPredictArray[19][2];
    int tPredictVMCount = predictVMCount;
    memcpy(tPredictArray,predictArray,(19*2)*4);
    if(tPredictVMCount > 0)
    {
        for(int i=1;i<=ecs.pFlavorTypeCount;i++)
        {
            server[i].push_back(phyServer(ecs.pFlavor[i].cpu,ecs.pFlavor[i].mem));
            pServerCount[i] ++;
        }
    }
    int flavorCount, bestChoiceIndex,tryCount = 0;
    bool isRestart = false, isGetBestChioce = false;
    double tempDiff, minDiff = DBL_MAX;

    while(tPredictVMCount)
    {
        minDiff = DBL_MAX;
        for(int i=MAX_FLAVOR_TYPE;i>0;i--)
        {
            flavorCount = tPredictArray[i][1];
            tryCount ++;
            while(flavorCount)
            {
                if(server[1][pServerCount[1]].usedCPU+ecs.vFlavor[i].cpu > server[1][0].MAX_CPU ||
                        server[1][pServerCount[1]].usedMEM+ecs.vFlavor[i].mem > server[1][0].MAX_MEM)
                {
                    if(tryCount >= MAX_FLAVOR_TYPE && tPredictVMCount > 0)
                    {
                        server[1][pServerCount[1]].isFull = true;
                        if(server[1][pServerCount[1]].getPercentageOfUsedCpu() > 0.95)
                            server[1][pServerCount[1]].isPerfectlyFull = true;
                        tryCount = 1;
                        server[1].push_back(phyServer(ecs.pFlavor[1].cpu,ecs.pFlavor[1].mem));
                        pServerCount[1] ++;
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
                    server[1][pServerCount[1]].usedCPU += ecs.vFlavor[i].cpu;
                    server[1][pServerCount[1]].usedMEM += ecs.vFlavor[i].mem;
                    tempDiff = fabs(server[1][pServerCount[1]].getPercentageOfUsedCpu()-server[1][pServerCount[1]].getPercentageOfUsedMem());
                    if(tempDiff <= minDiff)
                    {
                        minDiff = tempDiff;
                        bestChoiceIndex = i;
                        isGetBestChioce = true;
                    }
                    server[1][pServerCount[1]].usedCPU -= ecs.vFlavor[i].cpu;
                    server[1][pServerCount[1]].usedMEM -= ecs.vFlavor[i].mem;
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
            if(server[1][pServerCount[1]].usedCPU + ecs.vFlavor[bestChoiceIndex].cpu > server[1][0].MAX_CPU ||
               server[1][pServerCount[1]].usedMEM + ecs.vFlavor[bestChoiceIndex].cpu > server[1][0].MAX_MEM )
            {
                tryCount = 0;
                minDiff = DBL_MAX;
                isGetBestChioce = false;
                cout << "Get Bad Chioce!" << endl;
                break;
            }
            else
            {
                server[1][pServerCount[1]].addFlavor(ecs.vFlavor[bestChoiceIndex]);
                tPredictArray[bestChoiceIndex][1]--;
                tPredictVMCount--;
                tryCount = 0;
                minDiff = DBL_MAX;
                isGetBestChioce = false;
            }
//            cout << "Server[" << pServerCount[1] << "] add Flavor[" << bestChoiceFlavor << "]:" << endl;
//            cout << "Flavor[" << bestChoiceFlavor << "] count: " << tPredictArray[bestChoiceIndex][1] << endl;
//            cout << "server[" << pServerCount[1] << "] used CPU: " <<
//                    server[1][pServerCount[1]].usedCPU << " used MEM: " << server[1][pServerCount[1]].usedMEM
//                 << " server is full = " << server[1][pServerCount[1]].isFull << endl;
//            cout <<  "server[" << pServerCount[1] << "] used CPU: " << server[1][pServerCount[1]].getPercentageOfUsedCpu()*100 << "%, " <<
//                     "used MEM: " << server[1][pServerCount[1]].getPercentageOfUsedMem()*100 << "%" << endl;
//            cout << "=================" << endl;
//            system("pause");
        }
    }

//    cout << "Before, the predict data count:  VM count: " << predictVMCount << endl;
//    for(int i=1;i<=ecs.vFlavorTypeCount;i++)
//    {
//        cout << "Flavor" << predictArray[i][0] << "  Count: " << predictArray[i][1];
//        cout << endl;
//    }
//    cout << "=================" << endl;

//    if(SERVER_COUNT > 1)
//    {
//        int maxCount = 0;
//        int temp;
//        for(int i=1;i<=MAX_FLAVOR_TYPE;i++)
//        {
//            temp = server[SERVER_COUNT].flavorCount[serverInfo.flavorType[i]];
//            if(temp > maxCount)
//            {
//                maxCount = temp;
//                flavorType = serverInfo.flavorType[i];
//            }
//        }
//        if(maxCount < 2)
//        {
//            for(int i=1;i<=MAX_FLAVOR_TYPE;i++)
//            {
//                predictVMCount -= server[SERVER_COUNT].flavorCount[serverInfo.flavorType[i]];
//                predictArray[i][1] -=  server[SERVER_COUNT].flavorCount[serverInfo.flavorType[i]];
//            }
//            SERVER_COUNT--;
//        }
//        else
//        {
//            bool isThisFlavorCanPushIn;
//            for(int i=MAX_FLAVOR_TYPE;i>0;i--)
//            {
//                isThisFlavorCanPushIn = true;
//                while(isThisFlavorCanPushIn && !server[SERVER_COUNT].isFull)
//                {
//                    flavorType = serverInfo.flavorType[i];
//                    if(server[SERVER_COUNT].usedCPU+flavor[flavorType].cpu > MAX_CPU ||
//                            server[SERVER_COUNT].usedMEM+flavor[flavorType].mem > MAX_MEM)
//                    {
//                        if(i > 1)
//                           isThisFlavorCanPushIn = false;
//                        else
//                            server[SERVER_COUNT].isFull = true;
//                    }
//                    else
//                    {
//                        server[SERVER_COUNT].addFlavor(flavor[flavorType]);
//                        predictArray[i][1]++;
//                        predictVMCount++;
//                    }
//                }
//            }
//        }
//    }


//    cout << "After, the predict data count:  VM count: " << predictVMCount << endl;
//    for(int i=1;i<=serverInfo.flavorTypeCount;i++)
//    {
//        cout << "Flavor" << predictArray[i][0] << "  Count: " << predictArray[i][1];
//        cout << endl;
//    }
//    cout << "=================" << endl;

//    predictPhyServerCount = SERVER_COUNT;
    for(int i=1;i<=ecs.pFlavorTypeCount;i++)
    {
        if(server[i][pServerCount[i]].VMCount == 0)
        {
            pServerCount[i] -- ;
        }
    }
    cout << "DONE!" << endl;
}
