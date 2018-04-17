#include "predictModel.h"

void predictModel(int (&predictArray)[19][2], const DataLoader &ecs)
{
   double pArray_1[19][2];
   double pArray_2[19][2];
   double pArray_3[19][2];
   double pArray_4[19][2];
   for(int i=1;i<ecs.vFlavorTypeCount;i++)
   {
       pArray_1[i][0] = ecs.vFlavor[i].type;
       pArray_1[i][1] = 0.0;
       pArray_2[i][0] = ecs.vFlavor[i].type;
       pArray_2[i][1] = 0.0;
       pArray_3[i][0] = ecs.vFlavor[i].type;
       pArray_3[i][1] = 0.0;
       pArray_4[i][0] = ecs.vFlavor[i].type;
       pArray_4[i][1] = 0.0;
   }
   ESModel(pArray_1,ecs);
   twoDESModel(pArray_2,ecs);
   threeDESModel(pArray_3,ecs);
   linearModel(pArray_4,ecs);
   for(int i=1;i<=ecs.vFlavorTypeCount;i++)
   {
       predictArray[i][1] = int(ceil(pArray_1[i][1]*0.0+
                                     pArray_2[i][1]*0.0+
                                     pArray_3[i][1]*0.0+
                                     pArray_4[i][1]*1.0));
       if(predictArray[i][1] < 0)
           predictArray[i][1] = 0;
   }
}

// 复杂预测模型：预测每种flavor数量的数组，训练数据vector，训练数据的天数，预测的天数，物理服务器信息
void twoDESModel(double (&predictArray)[19][2], const DataLoader &ecs)
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
        predictArray[i][1] = (packedArray[i][packedArrayLength]);
    }
}

// 复杂预测模型：预测每种flavor数量的数组，训练数据vector，训练数据的天数，预测的天数，物理服务器信息
void ESModel(double (&predictArray)[19][2], const DataLoader &ecs)
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
    double a = 0.48;
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
    for(int i=1;i<=ecs.vFlavorTypeCount;i++)
        S1[i].resize(1+packedArrayLength);
    //    int initialPackSize = predictDaysCount;
    for(int i=1;i<=ecs.vFlavorTypeCount;i++)
    {
        S1[i][0] = packedArray[i][1];
        // 开始预测，预测已知数据
        for(int j=1;j<=packedArrayLength-packSize;j++)
        {
            S1[i][j] = a*packedArray[i][j]+(1-a)*S1[i][j-1];
        }
        S1[i][0] = packedArray[i][0];
        // 预测未知数据
        for(int j=1;j<=predictDaysCount;j++)
        {
            packedArray[i][packedArrayLength-packSize+j] = a*packedArray[i][packedArrayLength-packSize+j-1]+
                    (1-a)*S1[i][packedArrayLength-packSize+j-1];
            S1[i][packedArrayLength-packSize+j] = a*packedArray[i][packedArrayLength-packSize+j]+
                    (1-a)*S1[i][packedArrayLength-packSize+j-1];
        }
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
        predictArray[i][1] = (packedArray[i][packedArrayLength]*1.1);
}

// 复杂预测模型：预测每种flavor数量的数组，训练数据vector，训练数据的天数，预测的天数，物理服务器信息
void threeDESModel(double (&predictArray)[19][2], const DataLoader &ecs)
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
    double alpha = 0.05;
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
    vector<vector<double>> S3(1+ecs.vFlavorTypeCount);
    for(int i=1;i<=ecs.vFlavorTypeCount;i++)
    {
        S1[i].resize(1+packedArrayLength);
        S2[i].resize(1+packedArrayLength);
        S3[i].resize(1+packedArrayLength);
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
        S3[i][1] = S2[i][1];
        for(int j=2;j<=packedArrayLength-predictDaysCount;j++)
        {
            S3[i][j] = alpha*S2[i][j]+(1-alpha)*S3[i][j-1];
        }
        double a = 3*S1[i][packedArrayLength-predictDaysCount]-3*S2[i][packedArrayLength-predictDaysCount]+
                S3[i][packedArrayLength-predictDaysCount];
        double b = alpha/2/pow(1-alpha,2)*((6-5*alpha)*S1[i][packedArrayLength-predictDaysCount]-
                2*(5-4*alpha)*S2[i][packedArrayLength-predictDaysCount]+(4-3*alpha)*S3[i][packedArrayLength-predictDaysCount]);
        double c = pow(alpha,2)/2/pow(1-alpha,2)*(S1[i][packedArrayLength-predictDaysCount]-
                2*S2[i][packedArrayLength-predictDaysCount]+S3[i][packedArrayLength-predictDaysCount]);
        packedArray[i][packedArrayLength] = a+b*predictDaysCount+c*pow(predictDaysCount,2);
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
        predictArray[i][1] = (packedArray[i][packedArrayLength]*1.1);
    }
}

// 复杂预测模型：预测每种flavor数量的数组，训练数据vector，训练数据的天数，预测的天数，物理服务器信息
void linearModel(double (&predictArray)[19][2], const DataLoader &ecs)
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

    // 线性累加
    vector<vector<double>> accArray(1+ecs.vFlavorTypeCount);
    for(int i=1;i<=ecs.vFlavorTypeCount;i++)
    {
        accArray[i].resize(1+trainDataDaysCount+predictDaysCount);
        accArray[i][1] = trainDataArray[i][1];
        for(int j=2;j<=trainDataDaysCount;j++)
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
        double a = 0.8;
        vector<double> S(accArray[1].size());
        S.assign(S.size(),0.0);
        S[1] = accArray[i][1];
        for(int j=2;j<=trainDataDaysCount;j++)
            S[j] = a*accArray[i][j]+(1-a)*S[j-1];
        pArray[i].resize(1+trainDataDaysCount+predictDaysCount);
        while(it)
        {
            for(int j=0;j<predictDaysCount;j++)
                window[j] = delta*nD(j,double(predictDaysCount)/4)*alpha;
            // 预测
            temp = 0.0;
            for(int j=predictDaysCount+1;j<=trainDataDaysCount;j++)
            {
                for(int k=0;k<predictDaysCount;k++)
                    temp += S[j-predictDaysCount+k]*window[k];
                pArray[i][j] = temp;
                temp = 0.0;
            }
            // 计算误差
            temp = 0.0;
            for(int j=predictDaysCount+1;j<=trainDataDaysCount;j++)
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
        for(int j=predictDaysCount+1;j<=trainDataDaysCount+predictDaysCount;j++)
        {
            for(int k=0;k<predictDaysCount;k++)
                temp += S[j-predictDaysCount+k]*window[k];
            pArray[i][j] = temp;
            temp = 0.0;
            if(j > trainDataDaysCount)
                S[j] = pArray[i][j];
        }
        predictArray[i][1] = (pArray[i][trainDataDaysCount+predictDaysCount]-pArray[i][trainDataDaysCount+1]);
    }
}

double nD(double in, double sigma)
{
    return 1/sqrt(2*3.1415926)/sigma*exp(-pow(in,2)/2/pow(sigma,2));
}
