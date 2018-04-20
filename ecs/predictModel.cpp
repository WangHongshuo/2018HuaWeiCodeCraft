#include "predictModel.h"

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
        double alpha = 0.7;
        int it = 7000;
        double error = DBL_MAX;
        double a = 1.4;
        vector<double> S(accArray[1].size());
        S.assign(S.size(),0.0);
        S[1] = accArray[i][1];
        for(int j=2;j<=trainDataDayCount;j++)
            S[j] = a*accArray[i][j]+(1-a)*S[j-1];
        pArray[i].resize(1+ecs.predictEndIndex);
        while(it)
        {
            for(int j=0;j<predictDaysCount;j++)
                window[j] = delta*nD(j,double(predictDaysCount)*1.7)*alpha;
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
    }
}

double nD(double in, double sigma)
{
    return 1/sqrt(2*3.1415926)/sigma*exp(-pow(in,2)/2/pow(sigma,2));
}
