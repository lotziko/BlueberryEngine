#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class Object;
	class Camera;
	class GfxTexture;
	class Scene;
	class Material;

	class SceneObjectPicker
	{
	public:
		static void Initialize();
		static void Shutdown();

		static Object* Pick(Scene* scene, Camera* camera, int positionX, int positionY);
		static void DrawOutline(Scene* scene, Camera* camera, GfxTexture* colorRenderTarget, GfxTexture* depthStencilRenderTarget, const Vector2Int& viewport);

	private:
		static GfxTexture* s_SceneRenderTarget;
		static GfxTexture* s_SceneDepthStencil;
		static Material* s_SpriteObjectPickerMaterial;
		static Material* s_MeshObjectPickerMaterial;
		static Material* s_IconObjectPickerMaterial;
		static Material* s_ObjectPickerOutlineMaterial;
	};
}