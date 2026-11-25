#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class Scene;
	class Camera;
	class GfxTexture;

	class GizmoRenderer
	{
	public:
		static void GatherAndUpdate(Scene* scene);
		static void Draw(GfxTexture* target);
	};
}