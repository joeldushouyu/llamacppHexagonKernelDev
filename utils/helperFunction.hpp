#pragma once


#include <math.h>




inline float random_float(float min, float max) {
    return min + static_cast <float> (rand()) /( static_cast <float> ((float)RAND_MAX/(max-min)));
}


