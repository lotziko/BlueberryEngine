#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\ObjectPtr.h"

namespace Blueberry
{
	class Material;
	class GfxTexture;
	class Scene;
	class MeshRenderer;
	class SkyRenderer;
	class Camera;

	class MaterialPreview
	{
	public:
		void Draw(Material* material, GfxTexture* target);

	private:
		Scene* m_Scene;
		ObjectPtr<Material> m_Material;
		MeshRenderer* m_MeshRenderer;
		SkyRenderer* m_SkyRenderer;
		Camera* m_Camera;
	};
}