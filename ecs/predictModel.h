﻿#ifndef PREDICTMODEL_H
#define PREDICTMODEL_H
#include <fstream>
#include <iostream>
#include <float.h>
#include <vector>
#include <math.h>
#include <string.h>
#include <string>
#include "DataLoader.h"

void predictModel(int (&predictArray)[19][2], const DataLoader &ecs);
double nD(double in, double sigma);

#endif
