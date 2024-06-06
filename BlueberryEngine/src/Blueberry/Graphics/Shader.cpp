#include "bbpch.h"
#include "Shader.h"

#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Core\ClassDB.h"

#include "Blueberry\Graphics\DefaultTextures.h"

namespace Blueberry
{
	DATA_DEFINITION(TextureParameterData)
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

	const int& TextureParameterData::GetIndex() const
	{
		return m_Index;
	}

	void TextureParameterData::SetIndex(const int& index)
	{
		m_Index = index;
	}

	void TextureParameterData::BindProperties()
	{
		BEGIN_OBJECT_BINDING(TextureParameterData)
		BIND_FIELD(FieldInfo(TO_STRING(m_Name), &TextureParameterData::m_Name, BindingType::String))
		BIND_FIELD(FieldInfo(TO_STRING(m_DefaultTextureName), &TextureParameterData::m_DefaultTextureName, BindingType::String))
		BIND_FIELD(FieldInfo(TO_STRING(m_Index), &TextureParameterData::m_Index, BindingType::Int))
		END_OBJECT_BINDING()
	}

	const CullMode& ShaderData::GetCullMode() const
	{
		return m_CullMode;
	}

	void ShaderData::SetCullMode(const CullMode& cullMode)
	{
		m_CullMode = cullMode;
	}

	const BlendMode& ShaderData::GetBlendSrc() const
	{
		return m_SrcBlend;
	}

	void ShaderData::SetBlendSrc(const BlendMode& blendSrc)
	{
		m_SrcBlend = blendSrc;
	}

	const BlendMode& ShaderData::GetBlendDst() const
	{
		return m_DstBlend;
	}

	void ShaderData::SetBlendDst(const BlendMode& blendDst)
	{
		m_DstBlend = blendDst;
	}

	const ZWrite& ShaderData::GetZWrite() const
	{
		return m_ZWrite;
	}

	void ShaderData::SetZWrite(const ZWrite& zWrite)
	{
		m_ZWrite = zWrite;
	}

	const std::vector<DataPtr<TextureParameterData>>& ShaderData::GetTextureParameters() const
	{
		return m_TextureParameters;
	}

	void ShaderData::SetTextureParameters(const std::vector<DataPtr<TextureParameterData>>& parameters)
	{
		m_TextureParameters = parameters;
	}

	const Texture2D* ShaderData::GetDefaultTexture(const std::string& parameterName) const
	{
		for (auto& textureParameter : m_TextureParameters)
		{
			TextureParameterData* parameterData = textureParameter.Get();
			if (parameterData->GetName() == parameterName)
			{
				std::string name = parameterData->GetDefaultTextureName();
				if (name == "white")
				{
					return DefaultTextures::GetWhite();
				}
			}
		}
		return nullptr;
	}

	void ShaderData::BindProperties()
	{
		BEGIN_OBJECT_BINDING(ShaderData)
		BIND_FIELD(FieldInfo(TO_STRING(m_CullMode), &ShaderData::m_CullMode, BindingType::Enum))
		BIND_FIELD(FieldInfo(TO_STRING(m_SrcBlend), &ShaderData::m_SrcBlend, BindingType::Enum))
		BIND_FIELD(FieldInfo(TO_STRING(m_DstBlend), &ShaderData::m_DstBlend, BindingType::Enum))
		BIND_FIELD(FieldInfo(TO_STRING(m_ZWrite), &ShaderData::m_ZWrite, BindingType::Enum))
		BIND_FIELD(FieldInfo(TO_STRING(m_TextureParameters), &ShaderData::m_TextureParameters, BindingType::DataArray).SetObjectType(TextureParameterData::Type))
		END_OBJECT_BINDING()
	}

	const ShaderData* Shader::GetData() const
	{
		return m_Data.Get();
	}

	void Shader::Initialize(void* vertexData, void* pixelData)
	{
		GfxDevice::CreateShader(vertexData, pixelData, m_Shader);
	}

	void Shader::Initialize(void* vertexData, void* pixelData, const ShaderData& data)
	{
		GfxDevice::CreateShader(vertexData, pixelData, m_Shader);
		m_Data = new ShaderData(data);
	}

	Shader* Shader::Create(void* vertexData, void* pixelData)
	{
		Shader* shader = Object::Create<Shader>();
		shader->Initialize(vertexData, pixelData);
		return shader;
	}

	Shader* Shader::Create(void* vertexData, void* pixelData, const ShaderData& data)
	{
		Shader* shader = Object::Create<Shader>();
		shader->Initialize(vertexData, pixelData, data);
		return shader;
	}

	void Shader::BindProperties()
	{
		BEGIN_OBJECT_BINDING(Shader)
		BIND_FIELD(FieldInfo(TO_STRING(m_Data), &Shader::m_Data, BindingType::Data).SetObjectType(ShaderData::Type))
		END_OBJECT_BINDING()
	}
}