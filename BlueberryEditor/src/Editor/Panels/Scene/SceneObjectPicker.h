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
		SceneObjectPicker();

		void Pick(Scene* scene, BaseCamera& camera, const int& positionX, const int& positionY, const int& viewportWidth, const int& viewportHeight);
	private:
		GfxTexture* m_SceneRenderTarget;
		GfxTexture* m_StagingRenderTarget;
		Material* m_SpriteObjectPickerMaterial;
	};
}