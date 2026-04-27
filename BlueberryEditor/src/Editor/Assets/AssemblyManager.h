#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class AssemblyManager
	{
	public:
		static void Unload();
		static void Load();
		static String GetAssemblyDirectory();
		static bool CreateSolution();
		static bool BuildEditor(const bool& incrementCount = true);
		static bool BuildRuntime();
	};
}