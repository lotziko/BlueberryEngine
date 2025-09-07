#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\Object.h"

namespace Blueberry
{
	class Scene;
	class Camera;

	struct CalculationParams
	{
		int tileSize;
		float texelPerUnit;
		int samplePerTexel;
		int maxSize;
		bool denoise;
	};

	class LightmappingManager
	{
	public:
		static void Clear();
		static void Calculate(Scene* scene, Camera* camera, const Vector2Int& viewport, uint8_t* output);
		static void Calculate(Scene* scene, const CalculationParams& params, uint8_t*& output, Vector2Int& outputSize, List<Vector4>& chartScaleOffset, Dictionary<ObjectId, uint32_t>& chartInstanceOffset);
		static void Shutdown();
	};
}