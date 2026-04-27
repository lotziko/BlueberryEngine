#pragma once

namespace Blueberry
{
	class EngineLayer
	{
	public:
		static void Register();
		static void Initialize();
		static void Shutdown();
		static void Update();
	};
}