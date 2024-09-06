#pragma once

#include "Blueberry\Core\Object.h"
#include "Blueberry\Graphics\Enums.h"
#include "Blueberry\Graphics\GfxShader.h"
#include "Blueberry\Core\DataPtr.h"

namespace Blueberry
{
	class GfxShader;

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
	};

	class Texture2D;

	class ShaderData : public Data
	{
		DATA_DECLARATION(ShaderData)

	public:
		ShaderData() = default;
		virtual ~ShaderData() = default;

		const PassData* GetPass(const UINT& index) const;
		const UINT& GetPassCount() const;
		void SetPasses(const std::vector<DataPtr<PassData>>& passes);

		const std::vector<DataPtr<TextureParameterData>>& GetTextureParameters() const;
		void SetTextureParameters(const std::vector<DataPtr<TextureParameterData>>& parameters);

		static void BindProperties();

	private:
		std::vector<DataPtr<PassData>> m_Passes;
		std::vector<DataPtr<TextureParameterData>> m_TextureParameters;
	};

	struct VariantsData
	{
		std::vector<UINT> vertexShaderIndices;
		UINT geometryShaderIndex;
		std::vector<UINT> fragmentShaderIndices;

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

		static void BindProperties();

	private:
		const Shader::ShaderVariant GetVariant(const UINT& vertexKeywordFlags, const UINT& fragmentKeywordFlags);

	private:
		DataPtr<ShaderData> m_Data;

		std::vector<GfxVertexShader*> m_VertexShaders;
		GfxGeometryShader* m_GeometryShader;
		std::vector<GfxFragmentShader*> m_FragmentShaders;

		friend struct GfxDrawingOperation;
		friend class Material;
	};
}