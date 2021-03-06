﻿#include "predictModel.h"

void predictModel(int (&predictArray)[19][2], const DataLoader &ecs)
{
    // 将vector数据放入数组中
    // int[i][0]为flavor类型
    // int[i][1]~int[i][trainDataDayCount]为该flavor类型每个索引日期的数量
    int trainDataDayCount = ecs.trainDataDaysCount;
    int predictDaysCount = ecs.predictDaysCount;
    vector<vector<int>> trainDataArray(1+ecs.vFlavorTypeCount);
    for(int i=1;i<=ecs.vFlavorTypeCount;i++)
        trainDataArray[i].resize(1+ecs.predictEndIndex);
    for(int i=1;i<=ecs.vFlavorTypeCount;i++)
    {
        trainDataArray[i][0] = ecs.vFlavor[i].type;
        for(int j=1;j<=trainDataDayCount;j++)
            trainDataArray[i][j] = ecs.tData[j].flavorCount[ecs.vFlavor[i].type];
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
    // 数组形式的trainDataArray，int[i][0]为flavor类型，i取值为1~ecs.vFlavorTypeCount
    // int[i][1]~int[i][trainDataDayCount]为该flavor类型每个索引日期的数量
    // 训练数据天数trainDataDayCount，预测天数predictDaysCount
    //　输出参数：predictArray
    // predictArray[i][0]为flavor的类型，已经初始化
    // predictArray[i][1]为该类型的数量，需要输入，i的取值为1~ecs.vFlavorTypeCount
    // TODO

    // 线性累加
    vector<vector<double>> accArray(1+ecs.vFlavorTypeCount);
    for(int i=1;i<=ecs.vFlavorTypeCount;i++)
    {
        accArray[i].resize(1+ecs.predictEndIndex);
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
    vector<vector<double>> pArray(1+ecs.vFlavorTypeCount);
    for(int i=1;i<=ecs.vFlavorTypeCount;i++)
    {
        double tmpSum, tmpAvg, tmpSigma;
        tmpSum = 0.0;
        for(int j=1;j<=ecs.trainDataDaysCount;j++)
            tmpSum += trainDataArray[i][j];
        tmpAvg = tmpSum/double(trainDataDayCount);
        tmpSum = 0.0;
        for(int j=1;j<=ecs.trainDataDaysCount;j++)
            tmpSum += pow(double(trainDataArray[i][j])-tmpAvg,2);
        tmpSigma = sqrt(tmpSum/double(trainDataDayCount));
        cout << tmpSigma << endl;
        double alpha = 0.7;
        int it = 7000;
        double error = DBL_MAX;
        double a = 0.7;
        vector<double> S(accArray[1].size());
        S.assign(S.size(),0.0);
        S[1] = accArray[i][1];
        for(int j=2;j<=trainDataDayCount;j++)
            S[j] = a*accArray[i][j]+(1-a)*S[j-1];
        pArray[i].resize(1+ecs.predictEndIndex);
        while(it)
        {
            for(int j=0;j<predictDaysCount;j++)
                window[j] = delta*nD(j,double(predictDaysCount)*(1.5))*alpha;
            // 预测
            temp = 0.0;
            for(int j=predictDaysCount+1;j<=trainDataDayCount;j++)
            {
                for(int k=0;k<predictDaysCount;k++)
                    temp += S[j-predictDaysCount+k]*window[k];
                pArray[i][j] = temp;
                temp = 0.0;
            }
            // 计算误差
            temp = 0.0;
            for(int j=predictDaysCount+1;j<=trainDataDayCount;j++)
                temp += pow(S[j]-pArray[i][j],2);
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
        temp = 0.0;
        for(int j=predictDaysCount+1;j<=ecs.predictEndIndex;j++)
        {
            for(int k=0;k<predictDaysCount;k++)
                temp += S[j-predictDaysCount+k]*window[k];
            pArray[i][j] = temp;
            temp = 0.0;
            if(j > trainDataDayCount)
                S[j] = pArray[i][j];
        }
        predictArray[i][1] = ceil((pArray[i][ecs.predictEndIndex]-pArray[i][ecs.predictBeginIndex-1])*0.7);

        int trainDataSum = 0;
        for(int j=1;j<ecs.trainDataDaysCount;j++)
            trainDataSum += trainDataArray[i][j];
        if(predictArray[i][1] >= 100 && predictArray[i][1] > trainDataSum)
            predictArray[i][1] = ceil(double(trainDataSum)*0.9);

    }
    // 计算预测准确度
//    vector<int> realData = {0,21,40,1,12,29,2,1,33,8,1,10,11,0,5,0,1,4,0}; // ori group
    vector<int> realData = {0,3,1,4,1,31,9,6,69,16,1,9,0,0,23,3,0,1,0}; // 1-3 group
//    vector<int> realData = {0,6,14,9,1,82,37,8,80,18,1,15,21,5,0,1,0,0,0}; // 3-5 group
//    vector<int> realData = {0,18,36,7,9,36,22,15,152,32,4,28,36,0,13,6,0,3,0}; // 6-8 group
    double temp1 = 0.0,temp2 = 0.0, temp3 = 0.0;
    for(int i=1;i<=ecs.vFlavorTypeCount;i++)
    {
        temp1 += pow(double(predictArray[i][1]-realData[i]),2);
        temp2 += pow(double(realData[i]),2);
        temp3 += pow(double(predictArray[i][1]),2);
    }
    temp1 = sqrt(temp1/ecs.vFlavorTypeCount);
    temp2 = sqrt(temp2/ecs.vFlavorTypeCount);
    temp3 = sqrt(temp3/ecs.vFlavorTypeCount);
    double accuracy = (1-temp1/(temp2+temp3));
    cout << "Accurcy: " << accuracy << endl;
}

double nD(double in, double sigma)
{
    return 1/sqrt(2*3.1415926)/sigma*exp(-pow(in,2)/2/pow(sigma,2));
}
