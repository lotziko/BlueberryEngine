#pragma once

#include "Blueberry\Core\Object.h"
#include "Blueberry\Graphics\Enums.h"
#include "Blueberry\Graphics\GfxShader.h"
#include "Blueberry\Core\DataPtr.h"

namespace Blueberry
{
	class GfxShader;

	class KeywordDB
	{
	public:
		KeywordDB() = default;
		virtual ~KeywordDB() = default;

		const uint32_t& GetMask(const size_t& id);

	private:
		std::unordered_map<size_t, uint32_t> m_KeywordMask = {};
		uint32_t m_MaxMask = 1;
	};

	class TextureParameterData : public Data
	{
		DATA_DECLARATION(TextureParameterData)

	public:
		TextureParameterData() = default;
		virtual ~TextureParameterData() = default;

		const std::string& GetName() const;
		void SetName(const std::string& name);

		const std::string& GetDefaultTextureName() const;
		void SetDefaultTextureName(const std::string& name);

		static void BindProperties();

	private:
		std::string m_Name;
		std::string m_DefaultTextureName;
	};

	class PassData : public Data
	{
		DATA_DECLARATION(PassData)

	public:
		PassData() = default;
		virtual ~PassData() = default;

		const CullMode& GetCullMode() const;
		void SetCullMode(const CullMode& cullMode);

		const BlendMode& GetBlendSrcColor() const;
		const BlendMode& GetBlendSrcAlpha() const;
		void SetBlendSrc(const BlendMode& blendSrc);
		void SetBlendSrc(const BlendMode& blendSrcColor, const BlendMode& blendSrcAlpha);

		const BlendMode& GetBlendDstColor() const;
		const BlendMode& GetBlendDstAlpha() const;
		void SetBlendDst(const BlendMode& blendDst);
		void SetBlendDst(const BlendMode& blendDstColor, const BlendMode& blendDstAlpha);

		const ZTest& GetZTest() const;
		void SetZTest(const ZTest& zTest);

		const ZWrite& GetZWrite() const;
		void SetZWrite(const ZWrite& zWrite);

		const std::vector<std::string>& GetVertexKeywords() const;
		void SetVertexKeywords(const std::vector<std::string>& keywords);

		const std::vector<std::string>& GetFragmentKeywords() const;
		void SetFragmentKeywords(const std::vector<std::string>& keywords);

		const uint32_t& GetVertexOffset() const;
		void SetVertexOffset(const uint32_t& offset);

		const uint32_t& GetGeometryOffset() const;
		void SetGeometryOffset(const uint32_t& offset);

		const uint32_t& GetFragmentOffset() const;
		void SetFragmentOffset(const uint32_t& offset);

		static void BindProperties();

	private:
		CullMode m_CullMode = CullMode::Front;
		BlendMode m_SrcBlendColor = BlendMode::One;
		BlendMode m_DstBlendColor = BlendMode::Zero;
		BlendMode m_SrcBlendAlpha = BlendMode::One;
		BlendMode m_DstBlendAlpha = BlendMode::Zero;
		ZTest m_ZTest = ZTest::LessEqual;
		ZWrite m_ZWrite = ZWrite::On;
		std::vector<std::string> m_VertexKeywords;
		std::vector<std::string> m_FragmentKeywords;
		uint32_t m_VertexOffset;
		uint32_t m_GeometryOffset;
		uint32_t m_FragmentOffset;
	};

	class Texture2D;

	class ShaderData : public Data
	{
		DATA_DECLARATION(ShaderData)

	public:
		ShaderData() = default;
		virtual ~ShaderData() = default;

		const PassData* GetPass(const uint32_t& index) const;
		const uint32_t& GetPassCount() const;
		void SetPasses(const std::vector<PassData*>& passes);

		const std::vector<DataPtr<TextureParameterData>>& GetTextureParameters() const;
		void SetTextureParameters(const std::vector<DataPtr<TextureParameterData>>& parameters);

		static void BindProperties();

	private:
		std::vector<DataPtr<PassData>> m_Passes;
		std::vector<DataPtr<TextureParameterData>> m_TextureParameters;
	};

	struct VariantsData
	{
		std::vector<uint32_t> vertexShaderIndices;
		std::vector<uint32_t> geometryShaderIndices;
		std::vector<uint32_t> fragmentShaderIndices;

		std::vector<void*> shaders;
	};

	struct ShaderVariant
	{
		GfxVertexShader* vertexShader;
		GfxGeometryShader* geometryShader;
		GfxFragmentShader* fragmentShader;
	};

	class Shader : public Object
	{
		OBJECT_DECLARATION(Shader)
	
	public:

		Shader() = default;
		virtual ~Shader() = default;

		const ShaderData* GetData() const;

		void Initialize(const VariantsData& variantsData);
		void Initialize(const VariantsData& variantsData, const ShaderData& data);

		static Shader* Create(const VariantsData& variantsData, const ShaderData& shaderData);

		static void SetKeyword(const size_t& id, const bool& enabled);

		static void BindProperties();

	private:
		const Shader::ShaderVariant GetVariant(const uint32_t& vertexKeywordFlags, const uint32_t& fragmentKeywordFlags, const uint8_t& passIndex);

	private:
		DataPtr<ShaderData> m_Data;

		std::vector<GfxVertexShader*> m_VertexShaders;
		std::vector<GfxGeometryShader*> m_GeometryShaders;
		std::vector<GfxFragmentShader*> m_FragmentShaders;
		std::vector<std::tuple<uint32_t, uint32_t, uint32_t>> m_PassesOffsets;
		KeywordDB m_LocalKeywords = {};

		static std::unordered_set<size_t> s_ActiveKeywords;
		static uint32_t s_ActiveKeywordsMask;
		static KeywordDB s_GlobalKeywords;

		friend struct GfxDrawingOperation;
		friend class Material;
	};
}