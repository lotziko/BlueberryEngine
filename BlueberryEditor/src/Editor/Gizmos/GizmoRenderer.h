#pragma once

namespace Blueberry
{
	class Scene;
	class Camera;
	class GfxTexture;

	class GizmoRenderer
	{
	public:
		static void Draw(Scene* scene, Camera* camera, GfxTexture* target);
	};
}