#pragma once
#include "Blueberry\Core\ObjectPtr.h"

namespace Blueberry
{
	class Texture;
	class Shader;
	class GfxTexture;
	class ShaderOptions;
	enum class CullMode;
	enum class SurfaceType;

	class Material : public Object
	{
		OBJECT_DECLARATION(Material)

	public:
		Material() = default;

		static Material* Create(Shader* shader);

		void SetTexture(std::size_t id, Texture* texture);
		void SetTexture(std::string name, Texture* texture);

		void SetShader(Shader* shader);

		const ShaderOptions& GetShaderOptions();

		static void BindProperties();

	private:
		void FillGfxTextures();

	private:
		std::vector<std::pair<std::size_t, GfxTexture*>> m_GfxTextures;
		std::map<std::size_t, ObjectPtr<Texture>> m_Textures;
		ObjectPtr<Shader> m_Shader;
		/*CullMode m_CullMode = (CullMode)-1;
		SurfaceType m_SurfaceType = (SurfaceType)-1;*/

		friend struct GfxDrawingOperation;
	};
}