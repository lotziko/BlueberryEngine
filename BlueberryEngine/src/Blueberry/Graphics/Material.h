#pragma once
#include "Blueberry\Core\ObjectPtr.h"
#include "Blueberry\Graphics\Shader.h"
//#include "Blueberry\Graphics\GfxDrawingOperation.h"

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

		void SetTexture(size_t id, Texture* texture);
		void SetTexture(std::string name, Texture* texture);

		Shader* GetShader();
		void SetShader(Shader* shader);

		void ApplyProperties();

		const ShaderData* GetShaderData();
		DataList<TextureData>& GetTextureDatas();
		void AddTextureData(const TextureData& data);

		void SetKeyword(const std::string& keyword, const bool& enabled);
		const uint32_t& GetActiveKeywordsMask();

		const uint32_t& GetCRC();
		Texture* GetTexture(const size_t& id);

	private:
		void FillTextureMap();

	private:
		DataList<TextureData> m_Textures;
		List<std::string> m_ActiveKeywords;
		ObjectPtr<Shader> m_Shader;

		Dictionary<size_t, ObjectId> m_BindedTextures;

		uint32_t m_Crc = UINT32_MAX;
		uint32_t m_ActiveKeywordsMask = 0;

		friend struct GfxDrawingOperation;
		friend class GfxRenderStateCache;
	};
}