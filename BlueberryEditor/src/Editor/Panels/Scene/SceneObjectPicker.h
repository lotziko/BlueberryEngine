#pragma once

namespace Blueberry
{
	class BaseCamera;
	class GfxTexture;
	class Scene;
	class Material;

	class SceneObjectPicker
	{
	public:
		SceneObjectPicker(GfxTexture* depthStencilTexture);
		virtual ~SceneObjectPicker();

		void Pick(Scene* scene, BaseCamera& camera, const int& positionX, const int& positionY);
		void DrawOutline(Scene* scene, BaseCamera& camera, GfxTexture* renderTarget);
	private:
		GfxTexture* m_SceneRenderTarget = nullptr;
		GfxTexture* m_SceneDepthStencil = nullptr;
		GfxTexture* m_PixelRenderTarget = nullptr;
		Material* m_SpriteObjectPickerMaterial = nullptr;
		Material* m_MeshObjectPickerMaterial = nullptr;
		Material* m_ObjectPickerOutlineMaterial = nullptr;
	};
}