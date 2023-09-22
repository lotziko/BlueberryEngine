#pragma once

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
		Material(const Ref<Shader>& shader);

		static Ref<Material> Create(const Ref<Shader>& shader);

		void SetTexture(std::size_t id, const Ref<Texture>& texture);
		void SetTexture(std::string name, const Ref<Texture>& texture);

	private:
		void FillGfxTextures();

	private:
		std::vector<std::pair<std::size_t, GfxTexture*>> m_GfxTextures;
		std::map<std::size_t, Ref<Texture>> m_Textures;
		Ref<Shader> m_Shader;

		friend struct GfxDrawingOperation;
	};
}