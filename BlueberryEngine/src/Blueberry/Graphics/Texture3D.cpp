#include "Blueberry\Graphics\Texture3D.h"

#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\GfxTexture.h"
#include "Blueberry\Core\ClassDB.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Texture3D, Texture)
	{
		DEFINE_BASE_FIELDS(Texture3D, Texture)
		DEFINE_FIELD(Texture3D, m_Depth, BindingType::Int, FieldOptions())
	}

	Texture3D::~Texture3D()
	{
		if (m_Texture != nullptr)
		{
			delete m_Texture;
		}
	}

	uint32_t Texture3D::GetDepth() const
	{
		return m_Depth;
	}

	void Texture3D::Initialize(uint32_t width, uint32_t height, uint32_t depth, TextureFormat textureFormat)
	{
		m_Width = width;
		m_Height = height;
		m_Depth = depth;
		m_Format = textureFormat;
		if (m_Texture != nullptr)
		{
			delete m_Texture;
			m_Texture = nullptr;
		}
	}

	void Texture3D::SetData(uint8_t* data, size_t dataSize)
	{
		m_RawData.resize(dataSize);
		memcpy(m_RawData.data(), data, dataSize);
	}

	void Texture3D::Apply()
	{
		if (m_Texture != nullptr)
		{
			delete m_Texture;
			m_Texture = nullptr;
		}

		TextureProperties textureProperties = {};

		textureProperties.width = m_Width;
		textureProperties.height = m_Height;
		textureProperties.depth = m_Depth;
		textureProperties.data = m_RawData.data();
		textureProperties.dataSize = m_RawData.size();
		textureProperties.mipCount = m_MipCount;
		textureProperties.format = m_Format;
		textureProperties.dimension = TextureDimension::Texture3D;
		textureProperties.wrapMode = m_WrapMode;
		textureProperties.filterMode = m_FilterMode;

		GfxDevice::CreateTexture(textureProperties, m_Texture);
		IncrementUpdateCount();
		if (!m_IsReadable)
		{
			m_RawData.clear();
			m_RawData.shrink_to_fit();
		}
	}

	Texture3D* Texture3D::Create(uint32_t width, uint32_t height, uint32_t depth, TextureFormat textureFormat, WrapMode wrapMode, FilterMode filterMode)
	{
		Texture3D* texture = Object::Create<Texture3D>();
		texture->m_Width = width;
		texture->m_Height = height;
		texture->m_Depth = depth;
		texture->m_MipCount = 1;
		texture->m_Format = textureFormat;
		texture->m_WrapMode = wrapMode;
		texture->m_FilterMode = filterMode;
		return texture;
	}
}