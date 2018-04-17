#include "predictModel.h"

void predictModel(int (&predictArray)[19][2], const DataLoader &ecs)
{
    // 输出数据到文件
//    ofstream output("F:/MATLAB_project/HW/train.txt",ios_base::out);
//    for(int i=1;i<=ecs.vFlavorTypeCount;i++)
//    {
//        for(int j=1;j<=trainDataDaysCount+predictDaysCount;j++)
//        {
//            if(j <= trainDataDaysCount)
//                output << vTrainData[j].flavorCount[serverInfo.flavorType[i]] << " ";
//            else
//                output << '0' << " ";
//        }
//        output << '\n';
//    }
//    int tempWeek = vTrainData[1].dayOfWeek;
//    for(int j=1;j<=trainDataDaysCount+predictDaysCount;j++)
//    {
//        output << tempWeek << " ";
//        tempWeek++;
//        if(tempWeek > 7)
//            tempWeek = 1;
//    }
//    output.close();
//    system("pause");

    int trainDataDaysCount = ecs.trainDataDaysCount;
    int predictDaysCount = ecs.predictDaysCount;
    vector<vector<int>> trainDataArray(1+ecs.vFlavorTypeCount);
    for(int i=1;i<=ecs.vFlavorTypeCount;i++)
        trainDataArray[i].resize(1+trainDataDaysCount+predictDaysCount);
    for(int i=1;i<=ecs.vFlavorTypeCount;i++)
    {
        trainDataArray[i][0] = ecs.vFlavor[i].type;
        for(int j=1;j<=trainDataDaysCount;j++)
            trainDataArray[i][j] = ecs.tData[j].flavorCount[ecs.vFlavor[i].type];
    }

    // 线性累加
    vector<vector<double>> accArray(1+ecs.vFlavorTypeCount);
    for(int i=1;i<=ecs.vFlavorTypeCount;i++)
    {
        accArray[i].resize(1+trainDataDaysCount+predictDaysCount);
        accArray[i][1] = trainDataArray[i][1];
        for(int j=2;j<=trainDataDaysCount;j++)
            accArray[i][j] = accArray[i][j-1]+trainDataArray[i][j];
    }
    // 组建训练数据,索引从0开始，x只需建立一次
    int timeStep = predictDaysCount;
    vector<vector<double>> x(timeStep);
    for(uint i=0;i<x.size();i++)
        x[i].resize(accArray[1].size()-1);
    vector<vector<double>> y(1);
    for(int i=0;i<1;i++)
        y[i].resize(accArray[1].size()-1);
    vector<double> S(accArray[1].size()-1);

    GRU gru;
    // 隐藏层，训练天数，预测天数
    int hDim = ceil(double(trainDataDaysCount)/double(2));

    gru.setParameters(hDim,trainDataDaysCount,predictDaysCount,timeStep);
    int seed = time(0);
    cout << "Seed: " << seed << endl;
    gru.setRandomSeed(seed);
    // 循环训练所有数据

    // alpha
    double a = 0.6;
    for(int h=1;h<=ecs.vFlavorTypeCount;h++)
    {
        S[0] = 0.0;
        // ES
//        for(int i=1;i<=predictDaysCount;i++)
//            S[0] += accArray[h][i];
//        S[0] /= predictDaysCount;
        S[1] = accArray[h][1];
        for(int i=2;i<=trainDataDaysCount;i++)
            S[i] = a*accArray[h][i]+(1-a)*S[i-1];


        for(uint i=0;i<x.size();i++)
            x[i].assign(x[0].size(),0.0);
        y[0].assign(y[0].size(),0.0);

        for(int i=0;i<int(timeStep);i++)
        {
            for(int j=timeStep-i;j<=trainDataDaysCount;j++)
            {
                x[i][j-timeStep+i] = S[j];
            }
        }

        for(int i=timeStep+1;i<=trainDataDaysCount;i++)
        {
            y[0][i-timeStep-1] = S[i];
        }

        // x输入，y目标，步长，迭代次数，停止迭代的误差
        gru.setData(x,y,0.02,2000,0.01);
        if(h == 1)
            gru.initCell();
        gru.initCellValue();
        gru.startTrainning();

        predictArray[h][1] = ceil(gru.getPredictData()*0.9);
        if(predictArray[h][1] < 0)
            predictArray[h][1] = 0;
    }
    // 计算预测准确度
//    vector<int> realData = {0,19,20,2,10,32,8,3,45,14,0,8,8,11,2,2};
//    double temp1 = 0.0,temp2 = 0.0, temp3 = 0.0;
//    for(int i=1;i<=ecs.vFlavorTypeCount;i++)
//    {
//        temp1 += pow(double(predictArray[i][1]-realData[i]),2);
//        temp2 += pow(double(realData[i]),2);
//        temp3 += pow(double(predictArray[i][1]),2);
//    }
//    temp1 = sqrt(temp1/ecs.vFlavorTypeCount);
//    temp2 = sqrt(temp2/ecs.vFlavorTypeCount);
//    temp3 = sqrt(temp3/ecs.vFlavorTypeCount);
//    double accuracy = (1-temp1/(temp2+temp3));
//    cout << "Accurcy: " << accuracy << endl;
}
