#pragma once
#include "Blueberry\Core\WeakObjectPtr.h"

namespace Blueberry
{
	class Texture;
	class Shader;
	class GfxTexture;

	class Material : public Object
	{
		OBJECT_DECLARATION(Material)

	public:
		Material() = default;

		static Material* Create(Shader* shader);

		void SetTexture(std::size_t id, Texture* texture);
		void SetTexture(std::string name, Texture* texture);

		static void BindProperties();

	private:
		void FillGfxTextures();

	private:
		std::vector<std::pair<std::size_t, GfxTexture*>> m_GfxTextures;
		std::map<std::size_t, WeakObjectPtr<Texture>> m_Textures;
		WeakObjectPtr<Shader> m_Shader;

		friend struct GfxDrawingOperation;
	};
}