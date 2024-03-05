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

		void Pick(Scene* scene, BaseCamera& camera, const int& positionX, const int& positionY, const int& viewportWidth, const int& viewportHeight);
	private:
		GfxTexture* m_SceneRenderTarget = nullptr;
		GfxTexture* m_SceneDepthStencil = nullptr;
		GfxTexture* m_PixelRenderTarget = nullptr;
		Material* m_SpriteObjectPickerMaterial = nullptr;
	};
}