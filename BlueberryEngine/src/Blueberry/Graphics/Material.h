#pragma once

namespace Blueberry
{
	class Texture;
	class Shader;

	class Material : public Object
	{
		OBJECT_DECLARATION(Material)

	public:
		Material() = default;
		Material(const Ref<Shader>& shader);

		static Ref<Material> Create(const Ref<Shader>& shader);

		void SetTexture(const Ref<Texture>& texture);

	private:
		Ref<Texture> m_Texture;
		Ref<Shader> m_Shader;

		friend struct GfxDrawingOperation;
	};
}