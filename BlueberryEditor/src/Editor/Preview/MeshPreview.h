#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\ObjectPtr.h"

namespace Blueberry
{
	class Mesh;
	class GfxTexture;
	class Scene;
	class Material;
	class MeshRenderer;
	class Camera;

	class MeshPreview
	{
	public:
		void Draw(Mesh* mesh, GfxTexture* target);

	private:
		Scene* m_Scene;
		ObjectPtr<Mesh> m_Mesh;
		Material* m_MeshPreviewMaterial;
		MeshRenderer* m_Renderer;
		Camera* m_Camera;
	};
}