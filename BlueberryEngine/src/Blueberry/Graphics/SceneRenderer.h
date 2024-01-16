#pragma once

#include "Blueberry\Math\Math.h"

namespace Blueberry
{
	class Scene;
	class BaseCamera;

	class SceneRenderer
	{
	public:
		static void Draw(Scene* scene);
		static void Draw(Scene* scene, BaseCamera* camera);
	};
}