#ifndef PREDICTMODEL_H
#define PREDICTMODEL_H
#include <fstream>
#include <iostream>
#include <float.h>
#include <vector>
#include <math.h>
#include <string.h>
#include <string>
#include "GRU.h"
#include "DataLoader.h"

void predictModel(int (&predictArray)[19][2], const DataLoader &ecs);

#endif
