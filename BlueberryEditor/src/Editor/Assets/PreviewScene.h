#pragma once

namespace Blueberry
{
	class Material;
	class Mesh;
	class RenderTexture;
	class Scene;
	class MeshRenderer;
	class Camera;

	class PreviewScene
	{
	public:
		void Draw(Material* material, RenderTexture* target);
		void Draw(Mesh* mesh, RenderTexture* target);

	private:
		Scene* m_Scene;
		MeshRenderer* m_Renderer;
		Camera* m_Camera;
	};
}