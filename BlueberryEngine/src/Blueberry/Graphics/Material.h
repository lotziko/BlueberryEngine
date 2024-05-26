#pragma once
#include "Blueberry\Core\ObjectPtr.h"
#include "Blueberry\Graphics\Shader.h"

namespace Blueberry
{
	class Texture;
	class GfxTexture;
	enum class CullMode;
	enum class SurfaceType;
	
	class TextureData : public Data
	{
		DATA_DECLARATION(TextureData)

	public:
		static void BindProperties();

		const std::string& GetName();
		void SetName(const std::string& name);

		Texture* GetTexture();
		void SetTexture(Texture* texture);

	private:
		std::string m_Name;
		ObjectPtr<Texture> m_Texture;

		friend class Material;
	};

	// New material properties will appear here after saving them in the inspector and calling update
	class Material : public Object
	{
		OBJECT_DECLARATION(Material)

	public:
		Material() = default;
		virtual ~Material() = default;

		static Material* Create(Shader* shader);

		void SetTexture(std::size_t id, Texture* texture);
		void SetTexture(std::string name, Texture* texture);

		Shader* GetShader();
		void SetShader(Shader* shader);

		void ApplyShaderProperties();

		const ShaderData* GetShaderData();
		std::vector<DataPtr<TextureData>>& GetTextureDatas();
		void AddTextureData(TextureData* data);

		static void BindProperties();

		virtual void OnCreate() override final;

	private:
		void FillGfxTextures();

	private:
		std::unordered_map<std::size_t, GfxTexture*> m_GfxTextures;
		std::vector<DataPtr<TextureData>> m_Textures;
		ObjectPtr<Shader> m_Shader;

		friend struct GfxDrawingOperation;
	};
}