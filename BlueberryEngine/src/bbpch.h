#pragma once

#include <any>
#include <vector>
#include <map>
#include <set>
#include <unordered_set>
#include <queue>
#include <stack>
#include <string>
#include <array>
#include <bitset>
#include <iterator>
#include <sstream>
#include <tuple>
#include <regex>
#include <thread>
#include <mutex>

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\Memory.h"
#include "Blueberry\Core\Object.h"
#include "Blueberry\Core\Time.h"
#include "Blueberry\Math\Math.h"

#include <Windows.h>
#include "Blueberry\Tools\WindowsHelper.h"

#include "Blueberry\Logging\Log.h"
#include "Blueberry\Logging\TimeMeasurement.h"

#include "flathashmap\flat_hash_map.hpp"

#include <d3d11_4.h>
#include <dxgi1_6.h>
#include <d3dcompiler.h>

#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3d11.lib" )       // direct3D library
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "dxguid.lib")