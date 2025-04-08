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

	HashSet<size_t> Shader::s_ActiveKeywords = {};
	uint32_t Shader::s_ActiveKeywordsMask = 0;
	KeywordDB Shader::s_GlobalKeywords = {};

	const uint32_t& KeywordDB::GetMask(const size_t& id)
	{
		auto it = m_KeywordMask.find(id);
		if (it != m_KeywordMask.end())
		{
			return it->second;
		}
		uint32_t mask = m_MaxMask;
		m_MaxMask = m_MaxMask << 1;
		m_KeywordMask.insert({ id, mask });
		return mask;
	}

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

	const BlendMode& PassData::GetBlendSrcColor() const
	{
		return m_SrcBlendColor;
	}

	const BlendMode& PassData::GetBlendSrcAlpha() const
	{
		return m_SrcBlendAlpha;
	}

	void PassData::SetBlendSrc(const BlendMode& blendSrc)
	{
		m_SrcBlendColor = blendSrc;
		m_SrcBlendAlpha = blendSrc;
	}

	void PassData::SetBlendSrc(const BlendMode& blendSrcColor, const BlendMode& blendSrcAlpha)
	{
		m_SrcBlendColor = blendSrcColor;
		m_SrcBlendAlpha = blendSrcAlpha;
	}

	const BlendMode& PassData::GetBlendDstColor() const
	{
		return m_DstBlendColor;
	}

	const BlendMode& PassData::GetBlendDstAlpha() const
	{
		return m_DstBlendAlpha;
	}

	void PassData::SetBlendDst(const BlendMode& blendDst)
	{
		m_DstBlendColor = blendDst;
		m_DstBlendAlpha = blendDst;
	}

	void PassData::SetBlendDst(const BlendMode& blendDstColor, const BlendMode& blendDstAlpha)
	{
		m_DstBlendColor = blendDstColor;
		m_DstBlendAlpha = blendDstAlpha;
	}

	const ZTest& PassData::GetZTest() const
	{
		return m_ZTest;
	}

	void PassData::SetZTest(const ZTest& zTest)
	{
		m_ZTest = zTest;
	}

	const ZWrite& PassData::GetZWrite() const
	{
		return m_ZWrite;
	}

	void PassData::SetZWrite(const ZWrite& zWrite)
	{
		m_ZWrite = zWrite;
	}

	const List<std::string>& PassData::GetVertexKeywords() const
	{
		return m_VertexKeywords;
	}

	void PassData::SetVertexKeywords(const List<std::string>& keywords)
	{
		m_VertexKeywords = keywords;
	}

	const List<std::string>& PassData::GetFragmentKeywords() const
	{
		return m_FragmentKeywords;
	}

	void PassData::SetFragmentKeywords(const List<std::string>& keywords)
	{
		m_FragmentKeywords = keywords;
	}

	const uint32_t& PassData::GetVertexOffset() const
	{
		return m_VertexOffset;
	}

	void PassData::SetVertexOffset(const uint32_t& offset)
	{
		m_VertexOffset = offset;
	}

	const uint32_t& PassData::GetGeometryOffset() const
	{
		return m_GeometryOffset;
	}

	void PassData::SetGeometryOffset(const uint32_t& offset)
	{
		m_GeometryOffset = offset;
	}

	const uint32_t& PassData::GetFragmentOffset() const
	{
		return m_FragmentOffset;
	}

	void PassData::SetFragmentOffset(const uint32_t& offset)
	{
		m_FragmentOffset = offset;
	}

	void PassData::BindProperties()
	{
		BEGIN_OBJECT_BINDING(PassData)
		BIND_FIELD(FieldInfo(TO_STRING(m_CullMode), &PassData::m_CullMode, BindingType::Enum))
		BIND_FIELD(FieldInfo(TO_STRING(m_SrcBlendColor), &PassData::m_SrcBlendColor, BindingType::Enum))
		BIND_FIELD(FieldInfo(TO_STRING(m_SrcBlendAlpha), &PassData::m_SrcBlendAlpha, BindingType::Enum))
		BIND_FIELD(FieldInfo(TO_STRING(m_DstBlendColor), &PassData::m_DstBlendColor, BindingType::Enum))
		BIND_FIELD(FieldInfo(TO_STRING(m_DstBlendAlpha), &PassData::m_DstBlendAlpha, BindingType::Enum))
		BIND_FIELD(FieldInfo(TO_STRING(m_ZTest), &PassData::m_ZTest, BindingType::Enum))
		BIND_FIELD(FieldInfo(TO_STRING(m_ZWrite), &PassData::m_ZWrite, BindingType::Enum))
		BIND_FIELD(FieldInfo(TO_STRING(m_VertexKeywords), &PassData::m_VertexKeywords, BindingType::StringList))
		BIND_FIELD(FieldInfo(TO_STRING(m_FragmentKeywords), &PassData::m_FragmentKeywords, BindingType::StringList))
		BIND_FIELD(FieldInfo(TO_STRING(m_VertexOffset), &PassData::m_VertexOffset, BindingType::Int))
		BIND_FIELD(FieldInfo(TO_STRING(m_GeometryOffset), &PassData::m_GeometryOffset, BindingType::Int))
		BIND_FIELD(FieldInfo(TO_STRING(m_FragmentOffset), &PassData::m_FragmentOffset, BindingType::Int))
		END_OBJECT_BINDING()
	}

	const PassData* ShaderData::GetPass(const uint32_t& index) const
	{
		if (index < 0 || index >= m_Passes.size())
		{
			return nullptr;
		}
		return m_Passes[index].Get();
	}

	const uint32_t& ShaderData::GetPassCount() const
	{
		return m_Passes.size();
	}

	void ShaderData::SetPasses(const List<PassData*>& passes)
	{
		m_Passes.resize(passes.size());
		for (int i = 0; i < passes.size(); ++i)
		{
			m_Passes[i] = DataPtr<PassData>(passes[i]);
		}
	}

	const List<DataPtr<TextureParameterData>>& ShaderData::GetTextureParameters() const
	{
		return m_TextureParameters;
	}

	void ShaderData::SetTextureParameters(const List<DataPtr<TextureParameterData>>& parameters)
	{
		m_TextureParameters = parameters;
	}

	void ShaderData::BindProperties()
	{
		BEGIN_OBJECT_BINDING(ShaderData)
		BIND_FIELD(FieldInfo(TO_STRING(m_Passes), &ShaderData::m_Passes, BindingType::DataList).SetObjectType(PassData::Type))
		BIND_FIELD(FieldInfo(TO_STRING(m_TextureParameters), &ShaderData::m_TextureParameters, BindingType::DataList).SetObjectType(TextureParameterData::Type))
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

		int geometryShadersCount = variantsData.geometryShaderIndices.size();
		m_GeometryShaders.resize(geometryShadersCount);
		for (int i = 0; i < geometryShadersCount; ++i)
		{
			GfxGeometryShader* geometryShader = nullptr;
			uint32_t index = variantsData.geometryShaderIndices[i];
			if (index != -1)
			{
				GfxDevice::CreateGeometryShader(variantsData.shaders[index], geometryShader);
			}
			m_GeometryShaders[i] = geometryShader;
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

	void Shader::SetKeyword(const size_t& id, const bool& enabled)
	{
		if (enabled)
		{
			s_ActiveKeywords.insert(id);
		}
		else
		{
			s_ActiveKeywords.erase(id);
		}
		s_ActiveKeywordsMask = 0;
		for (auto keyword : s_ActiveKeywords)
		{
			s_ActiveKeywordsMask |= s_GlobalKeywords.GetMask(keyword);
		}
	}

	const uint32_t& Shader::GetActiveKeywordsMask()
	{
		return s_ActiveKeywordsMask;
	}

	void Shader::BindProperties()
	{
		BEGIN_OBJECT_BINDING(Shader)
		BIND_FIELD(FieldInfo(TO_STRING(m_Data), &Shader::m_Data, BindingType::Data).SetObjectType(ShaderData::Type))
		END_OBJECT_BINDING()
	}

	const ShaderVariant Shader::GetVariant(const uint32_t& vertexKeywordFlags, const uint32_t& fragmentKeywordFlags, const uint8_t& passIndex)
	{
		if (m_PassesOffsets.size() == 0)
		{
			ShaderData* data = m_Data.Get();
			m_PassesOffsets.resize(data->GetPassCount());
			for (uint32_t i = 0; i < m_PassesOffsets.size(); ++i)
			{
				const PassData* passData = data->GetPass(i);
				m_PassesOffsets[i] = std::make_tuple(passData->GetVertexOffset(), passData->GetGeometryOffset(), passData->GetFragmentOffset());
			}
		}

		auto offsets = m_PassesOffsets[passIndex];
		return { m_VertexShaders[std::get<0>(offsets) + vertexKeywordFlags], m_GeometryShaders[std::get<1>(offsets)], m_FragmentShaders[std::get<2>(offsets) + fragmentKeywordFlags] };
	}
}