#pragma once

namespace Blueberry
{
	class AssemblyManager
	{
	public:
		static void Unload();
		static void Load();
		static bool Build(const bool& incrementCount = true);
	};
}