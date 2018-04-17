#include "predictModel.h"

void predictModel(int (&predictArray)[19][2], const DataLoader &ecs)
{
    // 将vector数据放入数组中
    // int[i][0]为flavor类型
    // int[i][1]~int[i][trainDataDaysCount]为该flavor类型每个索引日期的数量
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

    // 输出用例（输出全部可输出数据）：
//    for(int i=1;i<=ecs.vFlavorTypeCount;i++)
//    {
//        for(int j=0;j<=trainDataDaysCount;j++)
//            cout << trainDataArray[i][j] << " ";
//        cout << endl;
//    }
//    cout << "=================" << endl;

    // 输出数据到文件
//    ofstream output("F:/MATLAB_project/HW/test.txt",ios_base::out);
//    for(int i=1;i<=ecs.vFlavorTypeCount;i++)
//    {
//        for(int j=0;j<=trainDataDaysCount;j++)
//            output << trainDataArray[i][j] << " ";
//        output << '\n';
//    }
//    system("pause");

    // 可用参数：
    // 数组形式的trainDataArray，int[i][0]为flavor类型，i取值为1~ecs.vFlavorTypeCount
    // int[i][1]~int[i][trainDataDaysCount]为该flavor类型每个索引日期的数量
    // 训练数据天数trainDataDayCount，预测天数predictDaysCount
    // 物理服务器信息serverInfo（flavor种类数量serverInfo.flavorTypeCount）
    //　输出参数：predictArray
    // predictArray[i][0]为flavor的类型，已经初始化
    // predictArray[i][1]为该类型的数量，需要输入，i的取值为1~ecs.vFlavorTypeCount
    // TODO

    // 指数平滑预测
    double alpha = 0.11;
    int dataLength = trainDataDaysCount+predictDaysCount;
    int packSize = predictDaysCount;
    int packedArrayLength = 1+dataLength-packSize;

    vector<vector <double>> packedArray(1+ecs.vFlavorTypeCount);
    for(int i=1;i<=ecs.vFlavorTypeCount;i++)
        packedArray[i].resize(1+packedArrayLength);

    // 以预测天数打包
    for(int i=1;i<=ecs.vFlavorTypeCount;i++)
    {
       packedArray[i][0] = trainDataArray[i][0];
    }
    //
    for(int i=1;i<=ecs.vFlavorTypeCount;i++)
    {
        for(int j=1;j<=packedArrayLength-packSize;j++)
        {
            for(int k=0;k<packSize;k++)
            {
                packedArray[i][j] += trainDataArray[i][j+k];
            }
        }
    }
    // 初始化需要预测的部分
    for(int i=1;i<=ecs.vFlavorTypeCount;i++)
    {
        for(int j=packedArrayLength-packSize+1;j<=packedArrayLength;j++)
            packedArray[i][j] = 0;
    }
    // 输出用例（输出全部可输出数据）：
//    cout << "Packed Array:" << endl;
//    for(int i=1;i<=ecs.vFlavorTypeCount;i++)
//    {
//        for(int j=0;j<=PackedTrainArrayLength;j++)
//            cout << packedTrainArray[i][j] << " ";
//        cout << endl;
//    }
//    cout << "=================" << endl;

    // 指数平滑法
    vector<vector <double>> S1(1+ecs.vFlavorTypeCount);
    vector<vector<double>> S2(1+ecs.vFlavorTypeCount);
    for(int i=1;i<=ecs.vFlavorTypeCount;i++)
    {
        S1[i].resize(1+packedArrayLength);
        S2[i].resize(1+packedArrayLength);
    }

//    int initialPackSize = predictDaysCount;
    for(int i=1;i<=ecs.vFlavorTypeCount;i++)
    {
        S1[i][1] = packedArray[i][1];
        // 开始预测，预测已知数据
        for(int j=2;j<=packedArrayLength-predictDaysCount;j++)
        {
            S1[i][j] = alpha*packedArray[i][j]+(1-alpha)*S1[i][j-1];
        }
        S1[i][0] = packedArray[i][0];
        S2[i][1] = S1[i][1];
        for(int j=2;j<=packedArrayLength-predictDaysCount;j++)
        {
            S2[i][j] = alpha*S1[i][j]+(1-alpha)*S2[i][j-1];
        }
        double a = 2*S1[i][packedArrayLength-predictDaysCount]-S2[i][packedArrayLength-predictDaysCount];
        double b = alpha/(1-alpha)*(S1[i][packedArrayLength-predictDaysCount]-S2[i][packedArrayLength-predictDaysCount]);
        packedArray[i][packedArrayLength] = a+b*predictDaysCount;
    }
    // 输出用例（输出全部可输出数据）：
//    cout << "S1[i] Array:" << endl;
//    for(int i=1;i<=ecs.vFlavorTypeCount;i++)
//    {
//        for(int j=0;j<=packedArrayLength;j++)
//            cout << S1[i][j] << " ";
//        cout << endl;
//    }
//    cout << "=================" << endl;
//    cout << "PackedArray[i]" << endl;
//    for(int i=1;i<=ecs.vFlavorTypeCount;i++)
//    {
//        for(int j=0;j<=packedArrayLength;j++)
//            cout << packedArray[i][j] << " ";
//        cout << endl;
//    }
//    cout << "=================" << endl;

    for(int i=1;i<=ecs.vFlavorTypeCount;i++)
    {
        predictArray[i][1] = ceil(packedArray[i][packedArrayLength]);
        if(predictArray[i][1] < 0)
            predictArray[i][1] = 0;
    }
}

