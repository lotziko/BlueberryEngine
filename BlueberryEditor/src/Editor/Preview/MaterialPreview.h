#pragma once

namespace Blueberry
{
	class Material;
	class RenderTexture;
	class Scene;
	class MeshRenderer;
	class Camera;

	class MaterialPreview
	{
	public:
		void Draw(Material* material, RenderTexture* target);

	private:
		Scene* m_Scene;
		MeshRenderer* m_Renderer;
		Camera* m_Camera;
	};
}