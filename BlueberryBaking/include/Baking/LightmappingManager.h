#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class Scene;
	class Camera;

	class LightmappingManager
	{
	public:
		static void Clear();
		static void Calculate(Scene* scene, Camera* camera, const Vector2Int& viewport, uint8_t* output);
		static void Shutdown();
	};
}