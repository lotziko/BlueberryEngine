#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class Scene;
	class ComputeShader;
	class GfxBuffer;

	class BakedLighting
	{
	public:
		static void Initialize();
		static void Shutdown();
		static void Apply(Scene* scene);

	private:
		static void ReleaseIfNeeded();

	private:
		static ComputeShader* s_FillProbeAtlasShader;
		static GfxBuffer* s_FillProbeAtlasData;
		static GfxBuffer* s_BricksData;
		static GfxBuffer* s_ColorsData;
	};
}