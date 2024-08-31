#include "bbpch.h"
#include "Shader.h"

#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Core\ClassDB.h"

#include "Blueberry\Graphics\DefaultTextures.h"

namespace Blueberry
{
	DATA_DEFINITION(TextureParameterData)
	DATA_DEFINITION(PassData)
	DATA_DEFINITION(ShaderData)
	OBJECT_DEFINITION(Object, Shader)

	const std::string& TextureParameterData::GetName() const
	{
		return m_Name;
	}

	void TextureParameterData::SetName(const std::string& name)
	{
		m_Name = name;
	}

	const std::string& TextureParameterData::GetDefaultTextureName() const
	{
		return m_DefaultTextureName;
	}

	void TextureParameterData::SetDefaultTextureName(const std::string& name)
	{
		m_DefaultTextureName = name;
	}

	void TextureParameterData::BindProperties()
	{
		BEGIN_OBJECT_BINDING(TextureParameterData)
		BIND_FIELD(FieldInfo(TO_STRING(m_Name), &TextureParameterData::m_Name, BindingType::String))
		BIND_FIELD(FieldInfo(TO_STRING(m_DefaultTextureName), &TextureParameterData::m_DefaultTextureName, BindingType::String))
		END_OBJECT_BINDING()
	}

	const CullMode& PassData::GetCullMode() const
	{
		return m_CullMode;
	}

	void PassData::SetCullMode(const CullMode& cullMode)
	{
		m_CullMode = cullMode;
	}

	const BlendMode& PassData::GetBlendSrc() const
	{
		return m_SrcBlend;
	}

	void PassData::SetBlendSrc(const BlendMode& blendSrc)
	{
		m_SrcBlend = blendSrc;
	}

	const BlendMode& PassData::GetBlendDst() const
	{
		return m_DstBlend;
	}

	void PassData::SetBlendDst(const BlendMode& blendDst)
	{
		m_DstBlend = blendDst;
	}

	const ZWrite& PassData::GetZWrite() const
	{
		return m_ZWrite;
	}

	void PassData::SetZWrite(const ZWrite& zWrite)
	{
		m_ZWrite = zWrite;
	}

	const std::vector<std::string>& PassData::GetVertexKeywords() const
	{
		return m_VertexKeywords;
	}

	void PassData::SetVertexKeywords(const std::vector<std::string>& keywords)
	{
		m_VertexKeywords = keywords;
	}

	const std::vector<std::string>& PassData::GetFragmentKeywords() const
	{
		return m_FragmentKeywords;
	}

	void PassData::SetFragmentKeywords(const std::vector<std::string>& keywords)
	{
		m_FragmentKeywords = keywords;
	}

	void PassData::BindProperties()
	{
		BEGIN_OBJECT_BINDING(PassData)
		BIND_FIELD(FieldInfo(TO_STRING(m_CullMode), &PassData::m_CullMode, BindingType::Enum))
		BIND_FIELD(FieldInfo(TO_STRING(m_SrcBlend), &PassData::m_SrcBlend, BindingType::Enum))
		BIND_FIELD(FieldInfo(TO_STRING(m_DstBlend), &PassData::m_DstBlend, BindingType::Enum))
		BIND_FIELD(FieldInfo(TO_STRING(m_ZWrite), &PassData::m_ZWrite, BindingType::Enum))
		BIND_FIELD(FieldInfo(TO_STRING(m_VertexKeywords), &PassData::m_VertexKeywords, BindingType::StringArray))
		BIND_FIELD(FieldInfo(TO_STRING(m_FragmentKeywords), &PassData::m_FragmentKeywords, BindingType::StringArray))
		END_OBJECT_BINDING()
	}

	const PassData* ShaderData::GetPass(const UINT& index) const
	{
		if (index < 0 || index > m_Passes.size())
		{
			return nullptr;
		}
		return m_Passes[index].Get();
	}

	const UINT& ShaderData::GetPassCount() const
	{
		return m_Passes.size();
	}

	void ShaderData::SetPasses(const std::vector<DataPtr<PassData>>& passes)
	{
		m_Passes = passes;
	}

	const std::vector<DataPtr<TextureParameterData>>& ShaderData::GetTextureParameters() const
	{
		return m_TextureParameters;
	}

	void ShaderData::SetTextureParameters(const std::vector<DataPtr<TextureParameterData>>& parameters)
	{
		m_TextureParameters = parameters;
	}

	void ShaderData::BindProperties()
	{
		BEGIN_OBJECT_BINDING(ShaderData)
		BIND_FIELD(FieldInfo(TO_STRING(m_Passes), &ShaderData::m_Passes, BindingType::DataArray).SetObjectType(PassData::Type))
		BIND_FIELD(FieldInfo(TO_STRING(m_TextureParameters), &ShaderData::m_TextureParameters, BindingType::DataArray).SetObjectType(TextureParameterData::Type))
		END_OBJECT_BINDING()
	}

	const ShaderData* Shader::GetData() const
	{
		return m_Data.Get();
	}

	void Shader::Initialize(const VariantsData& variantsData)
	{
		int vertexShadersCount = variantsData.vertexShaderIndices.size();
		m_VertexShaders.resize(vertexShadersCount);
		for (int i = 0; i < vertexShadersCount; ++i)
		{
			GfxVertexShader* vertexShader;
			GfxDevice::CreateVertexShader(variantsData.shaders[variantsData.vertexShaderIndices[i]], vertexShader);
			m_VertexShaders[i] = vertexShader;
		}

		if (variantsData.geometryShaderIndex != -1)
		{
			GfxDevice::CreateGeometryShader(variantsData.shaders[variantsData.geometryShaderIndex], m_GeometryShader);
		}

		int fragmentShadersCount = variantsData.fragmentShaderIndices.size();
		m_FragmentShaders.resize(fragmentShadersCount);
		for (int i = 0; i < fragmentShadersCount; ++i)
		{
			GfxFragmentShader* fragmentShader;
			GfxDevice::CreateFragmentShader(variantsData.shaders[variantsData.fragmentShaderIndices[i]], fragmentShader);
			m_FragmentShaders[i] = fragmentShader;
		}
	}

	void Shader::Initialize(const VariantsData& variantsData, const ShaderData& data)
	{
		Initialize(variantsData);
		m_Data = new ShaderData(data);
	}

	Shader* Shader::Create(const VariantsData& variantsData, const ShaderData& shaderData)
	{
		Shader* shader = Object::Create<Shader>();
		shader->Initialize(variantsData, shaderData);
		return shader;
	}

	void Shader::BindProperties()
	{
		BEGIN_OBJECT_BINDING(Shader)
		BIND_FIELD(FieldInfo(TO_STRING(m_Data), &Shader::m_Data, BindingType::Data).SetObjectType(ShaderData::Type))
		END_OBJECT_BINDING()
	}

	const ShaderVariant Shader::GetVariant(const UINT& vertexKeywordFlags, const UINT& fragmentKeywordFlags)
	{
		return { m_VertexShaders[vertexKeywordFlags], m_GeometryShader, m_FragmentShaders[fragmentKeywordFlags] };
	}
}