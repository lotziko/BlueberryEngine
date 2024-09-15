#pragma once
#include "Blueberry\Core\ObjectPtr.h"
#include "Blueberry\Graphics\Shader.h"
#include "Blueberry\Graphics\GfxDrawingOperation.h"

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

		void ApplyProperties();

		const ShaderData* GetShaderData();
		std::vector<DataPtr<TextureData>>& GetTextureDatas();
		void AddTextureData(TextureData* data);

		void SetKeyword(const std::string& keyword, const bool& enabled);

		GfxRenderState* GetState(const uint8_t& passIndex); // TODO pass + global keywords
		const uint32_t& GetCRC();

		static void BindProperties();

	private:
		void FillTextureMap();

	private:
		std::vector<DataPtr<TextureData>> m_Textures;
		std::vector<std::string> m_ActiveKeywords;
		ObjectPtr<Shader> m_Shader;

		std::unordered_map<std::size_t, ObjectId> m_TextureMap;
		std::vector<GfxRenderState> m_PassCache;

		uint32_t m_Crc = -1;

		friend struct GfxDrawingOperation;
	};
}