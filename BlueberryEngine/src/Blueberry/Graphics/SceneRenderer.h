#pragma once

#include "Blueberry\Math\Math.h"

namespace Blueberry
{
	class Scene;
	class Camera;

	class SceneRenderer
	{
	public:
		static void Draw(Scene* scene);
		static void Draw(Scene* scene, Camera* camera);
		static void Draw(Scene* scene, const Matrix& viewMatrix, const Matrix& projectionMatrix);
	};
}