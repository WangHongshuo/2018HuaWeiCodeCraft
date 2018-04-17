#ifndef __ROUTE_H__
#define __ROUTE_H__

#include <fstream>
#include <iostream>
#include <float.h>
#include <vector>
#include <math.h>
#include <string.h>
#include <string>
#include "DataLoader.h"
#include "predictModel.h"
#include "allocateModel.h"
#include <float.h>
#include "lib_io.h"

using namespace std;

void predict_server(char * info[MAX_INFO_NUM], char * data[MAX_DATA_NUM], int data_num, char * filename);

#endif
