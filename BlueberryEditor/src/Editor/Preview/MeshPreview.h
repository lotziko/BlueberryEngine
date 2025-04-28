#pragma once

namespace Blueberry
{
	class Mesh;
	class RenderTexture;
	class Scene;
	class Material;
	class MeshRenderer;
	class Camera;

	class MeshPreview
	{
	public:
		void Draw(Mesh* mesh, RenderTexture* target);

	private:
		Scene* m_Scene;
		Material* m_MeshPreviewMaterial;
		MeshRenderer* m_Renderer;
		Camera* m_Camera;
	};
}