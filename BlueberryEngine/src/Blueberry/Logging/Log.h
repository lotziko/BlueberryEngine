#pragma once

#include "Logger.h"
#include <iostream>

namespace Blueberry
{
#define BB_INITIALIZE_LOG() //freopen("CONOUT$", "w", stdout);
#define BB_INFO(...) //std::cout << __VA_ARGS__ << std::endl;
#define BB_ERROR(...) //std::cout << __VA_ARGS__ << std::endl;
#define BB_WARNING(...) //std::cout << __VA_ARGS__ << std::endl;
}