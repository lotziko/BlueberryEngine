#pragma once

namespace Blueberry
{
	class Material;
	class GfxTexture;
	class Scene;
	class MeshRenderer;
	class Camera;

	class MaterialPreview
	{
	public:
		void Draw(Material* material, GfxTexture* target);

	private:
		Scene* m_Scene;
		MeshRenderer* m_Renderer;
		Camera* m_Camera;
	};
}