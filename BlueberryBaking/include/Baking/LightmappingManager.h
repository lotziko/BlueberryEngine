#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\Object.h"

namespace Blueberry
{
	class Scene;
	class Camera;

	class LightmappingManager
	{
	public:
		static void Clear();
		static void Calculate(Scene* scene, Camera* camera, const Vector2Int& viewport, uint8_t* output);
		static void Calculate(Scene* scene, const Vector2Int& tileSize, uint8_t*& output, Vector2Int& outputSize, List<Vector4>& chartScaleOffset, Dictionary<ObjectId, uint32_t>& chartInstanceOffset);
		static void Shutdown();
	};
}