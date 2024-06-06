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

		const int& GetIndex() const;
		void SetIndex(const int& index);

		static void BindProperties();

	private:
		std::string m_Name;
		std::string m_DefaultTextureName;
		int m_Index;
	};

	class Texture2D;

	class ShaderData : public Data
	{
		DATA_DECLARATION(ShaderData)

	public:
		ShaderData() = default;
		virtual ~ShaderData() = default;

		const CullMode& GetCullMode() const;
		void SetCullMode(const CullMode& cullMode);

		const BlendMode& GetBlendSrc() const;
		void SetBlendSrc(const BlendMode& blendSrc);

		const BlendMode& GetBlendDst() const;
		void SetBlendDst(const BlendMode& blendDst);

		const ZWrite& GetZWrite() const;
		void SetZWrite(const ZWrite& zWrite);

		const std::vector<DataPtr<TextureParameterData>>& GetTextureParameters() const;
		void SetTextureParameters(const std::vector<DataPtr<TextureParameterData>>& parameters);

		const Texture2D* GetDefaultTexture(const std::string& parameterName) const;

		static void BindProperties();

	private:
		CullMode m_CullMode = CullMode::Front;
		BlendMode m_SrcBlend = BlendMode::One;
		BlendMode m_DstBlend = BlendMode::Zero;
		ZWrite m_ZWrite = ZWrite::On;
		std::vector<DataPtr<TextureParameterData>> m_TextureParameters;
	};

	class Shader : public Object
	{
		OBJECT_DECLARATION(Shader)

	public:
		Shader() = default;
		virtual ~Shader() = default;

		const ShaderData* GetData() const;

		void Initialize(void* vertexData, void* pixelData);
		void Initialize(void* vertexData, void* pixelData, const ShaderData& data);
		static Shader* Create(void* vertexData, void* pixelData);
		static Shader* Create(void* vertexData, void* pixelData, const ShaderData& data);

		static void BindProperties();

	private:
		GfxShader* m_Shader;
		DataPtr<ShaderData> m_Data;

		friend struct GfxDrawingOperation;
	};
}