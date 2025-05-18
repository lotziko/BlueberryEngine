#include "Blueberry\Graphics\Shader.h"

#include "Blueberry\Core\ObjectDB.h"
#include "Blueberry\Core\Notifyable.h"
#include "Blueberry\Core\ClassDB.h"
#include "..\Graphics\GfxDevice.h"
#include "..\Graphics\GfxShader.h"
#include "..\Graphics\DefaultTextures.h"

namespace Blueberry
{
	DATA_DEFINITION(PropertyData)
	{
		DEFINE_FIELD(PropertyData, m_Name, BindingType::String, {})
		DEFINE_FIELD(PropertyData, m_Type, BindingType::Enum, {})
		DEFINE_FIELD(PropertyData, m_DefaultTextureName, BindingType::String, {})
		DEFINE_FIELD(PropertyData, m_TextureDimension, BindingType::Enum, {})
	}

	DATA_DEFINITION(PassData)
	{
		DEFINE_FIELD(PassData, m_CullMode, BindingType::Enum, {})
		DEFINE_FIELD(PassData, m_SrcBlendColor, BindingType::Enum, {})
		DEFINE_FIELD(PassData, m_SrcBlendAlpha, BindingType::Enum, {})
		DEFINE_FIELD(PassData, m_DstBlendColor, BindingType::Enum, {})
		DEFINE_FIELD(PassData, m_DstBlendAlpha, BindingType::Enum, {})
		DEFINE_FIELD(PassData, m_ZTest, BindingType::Enum, {})
		DEFINE_FIELD(PassData, m_ZWrite, BindingType::Enum, {})
		DEFINE_FIELD(PassData, m_VertexKeywords, BindingType::StringList, {})
		DEFINE_FIELD(PassData, m_FragmentKeywords, BindingType::StringList, {})
		DEFINE_FIELD(PassData, m_VertexOffset, BindingType::Int, {})
		DEFINE_FIELD(PassData, m_GeometryOffset, BindingType::Int, {})
		DEFINE_FIELD(PassData, m_FragmentOffset, BindingType::Int, {})
	}

	DATA_DEFINITION(ShaderData)
	{
		DEFINE_FIELD(ShaderData, m_Passes, BindingType::DataList, FieldOptions().SetObjectType(PassData::Type))
		DEFINE_FIELD(ShaderData, m_Properties, BindingType::DataList, FieldOptions().SetObjectType(PropertyData::Type))
	}

	OBJECT_DEFINITION(Shader, Object)
	{
		DEFINE_BASE_FIELDS(Shader, Object)
		DEFINE_FIELD(Shader, m_Data, BindingType::Data, FieldOptions().SetObjectType(ShaderData::Type))
	}

	HashSet<size_t> Shader::s_ActiveKeywords = {};
	uint32_t Shader::s_ActiveKeywordsMask = 0;
	KeywordDB Shader::s_GlobalKeywords = {};

	const uint32_t KeywordDB::GetMask(const size_t& id)
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

	const String& PropertyData::GetName() const
	{
		return m_Name;
	}

	void PropertyData::SetName(const String& name)
	{
		m_Name = name;
	}

	const PropertyData::PropertyType& PropertyData::GetType() const
	{
		return m_Type;
	}

	void PropertyData::SetType(const PropertyType& type)
	{
		m_Type = type;
	}

	const String& PropertyData::GetDefaultTextureName() const
	{
		return m_DefaultTextureName;
	}

	void PropertyData::SetDefaultTextureName(const String& name)
	{
		m_DefaultTextureName = name;
	}

	const TextureDimension& PropertyData::GetTextureDimension() const
	{
		return m_TextureDimension;
	}

	void PropertyData::SetTextureDimension(const TextureDimension& dimension)
	{
		m_TextureDimension = dimension;
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

	const List<String>& PassData::GetVertexKeywords() const
	{
		return m_VertexKeywords;
	}

	void PassData::SetVertexKeywords(const List<String>& keywords)
	{
		m_VertexKeywords = keywords;
	}

	const List<String>& PassData::GetFragmentKeywords() const
	{
		return m_FragmentKeywords;
	}

	void PassData::SetFragmentKeywords(const List<String>& keywords)
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

	const PassData& ShaderData::GetPass(const uint32_t& index) const
	{
		return m_Passes[index];
	}

	const size_t ShaderData::GetPassCount() const
	{
		return m_Passes.size();
	}

	void ShaderData::SetPasses(const DataList<PassData>& passes)
	{
		m_Passes = passes;
	}

	const DataList<PropertyData>& ShaderData::GetProperties() const
	{
		return m_Properties;
	}

	void ShaderData::SetProperties(const DataList<PropertyData>& properties)
	{
		m_Properties = properties;
	}

	const ShaderData& Shader::GetData() const
	{
		return m_Data;
	}

	void Shader::Initialize(const VariantsData& variantsData)
	{
		if (m_VertexShaders.size() > 0)
		{
			for (auto it = m_VertexShaders.begin(); it < m_VertexShaders.end(); ++it)
			{
				delete *it;
			}
			m_VertexShaders.clear();
		}
		if (m_GeometryShaders.size() > 0)
		{
			for (auto it = m_GeometryShaders.begin(); it < m_GeometryShaders.end(); ++it)
			{
				delete *it;
			}
			m_GeometryShaders.clear();
		}
		if (m_FragmentShaders.size() > 0)
		{
			for (auto it = m_FragmentShaders.begin(); it < m_FragmentShaders.end(); ++it)
			{
				delete *it;
			}
			m_FragmentShaders.clear();
		}
		if (m_PassesOffsets.size() > 0)
		{
			m_PassesOffsets.clear();
		}

		size_t vertexShadersCount = variantsData.vertexShaderIndices.size();
		m_VertexShaders.resize(vertexShadersCount);
		for (size_t i = 0; i < vertexShadersCount; ++i)
		{
			GfxVertexShader* vertexShader;
			GfxDevice::CreateVertexShader(variantsData.shaders[variantsData.vertexShaderIndices[i]], vertexShader);
			m_VertexShaders[i] = vertexShader;
		}

		size_t geometryShadersCount = variantsData.geometryShaderIndices.size();
		m_GeometryShaders.resize(geometryShadersCount);
		for (size_t i = 0; i < geometryShadersCount; ++i)
		{
			GfxGeometryShader* geometryShader = nullptr;
			uint32_t index = variantsData.geometryShaderIndices[i];
			if (index != -1)
			{
				GfxDevice::CreateGeometryShader(variantsData.shaders[index], geometryShader);
			}
			m_GeometryShaders[i] = geometryShader;
		}

		size_t fragmentShadersCount = variantsData.fragmentShaderIndices.size();
		m_FragmentShaders.resize(fragmentShadersCount);
		for (size_t i = 0; i < fragmentShadersCount; ++i)
		{
			GfxFragmentShader* fragmentShader;
			GfxDevice::CreateFragmentShader(variantsData.shaders[variantsData.fragmentShaderIndices[i]], fragmentShader);
			m_FragmentShaders[i] = fragmentShader;
		}
	}

	void Shader::Initialize(const VariantsData& variantsData, const ShaderData& data)
	{
		Initialize(variantsData);
		m_Data = data;
	}

	Shader* Shader::Create(const VariantsData& variantsData, const ShaderData& shaderData, Shader* existingShader)
	{
		Shader* shader = nullptr;
		if (existingShader != nullptr)
		{
			shader = existingShader;
			shader->IncrementUpdateCount();
		}
		else
		{
			shader = Object::Create<Shader>();
		}
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

	const ShaderVariant Shader::GetVariant(const uint32_t& vertexKeywordFlags, const uint32_t& fragmentKeywordFlags, const uint8_t& passIndex)
	{
		if (m_PassesOffsets.size() == 0)
		{
			m_PassesOffsets.resize(m_Data.GetPassCount());
			for (uint32_t i = 0; i < m_PassesOffsets.size(); ++i)
			{
				auto& passData = m_Data.GetPass(i);
				m_PassesOffsets[i] = std::make_tuple(passData.GetVertexOffset(), passData.GetGeometryOffset(), passData.GetFragmentOffset());
			}
		}

		auto offsets = m_PassesOffsets[passIndex];
		return { m_VertexShaders[std::get<0>(offsets) + vertexKeywordFlags], m_GeometryShaders[std::get<1>(offsets)], m_FragmentShaders[std::get<2>(offsets) + fragmentKeywordFlags] };
	}

	void Shader::IncrementUpdateCount()
	{
		++m_UpdateCount;
		for (auto dependency : m_Dependencies)
		{
			Object* object = ObjectDB::GetObject(dependency);
			if (object != nullptr)
			{
				dynamic_cast<Notifyable*>(object)->OnNotify();
			}
		}
	}
}