#pragma once

namespace Blueberry
{
	struct CullingResults;
	class ShadowAtlas;

	class RealtimeLights
	{
	public:
		static void PrepareShadows(CullingResults& results, ShadowAtlas* atlas);
		static void BindLights(CullingResults& results, ShadowAtlas* atlas);
	};
}