#pragma once

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
		SceneObjectPicker();
		virtual ~SceneObjectPicker();

		Object* Pick(Scene* scene, Camera* camera, const int& positionX, const int& positionY);
		void DrawOutline(Scene* scene, Camera* camera, GfxTexture* renderTarget);
	private:
		GfxTexture* m_SceneRenderTarget = nullptr;
		GfxTexture* m_SceneDepthStencil = nullptr;
		Material* m_SpriteObjectPickerMaterial = nullptr;
		Material* m_MeshObjectPickerMaterial = nullptr;
		Material* m_ObjectPickerOutlineMaterial = nullptr;
	};
}