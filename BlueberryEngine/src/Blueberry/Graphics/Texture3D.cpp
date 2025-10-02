#include "Blueberry\Graphics\Texture3D.h"

#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\GfxTexture.h"
#include "Blueberry\Core\ClassDB.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Texture3D, Texture)
	{
		DEFINE_BASE_FIELDS(Texture3D, Texture)
		DEFINE_FIELD(Texture3D, m_Depth, BindingType::Int, {})
	}

	Texture3D::~Texture3D()
	{
		if (m_Texture != nullptr)
		{
			delete m_Texture;
		}
	}

	const uint32_t& Texture3D::GetDepth()
	{
		return m_Depth;
	}

	void Texture3D::SetData(uint8_t* data, const size_t& dataSize)
	{
		m_RawData.resize(dataSize);
		memcpy(m_RawData.data(), data, dataSize);
	}

	void Texture3D::Apply()
	{
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
	}

	Texture3D* Texture3D::Create(const uint32_t& width, const uint32_t& height, const uint32_t& depth, const TextureFormat& textureFormat, const WrapMode& wrapMode, const FilterMode& filterMode, Texture3D* existingTexture)
	{
		Texture3D* texture = nullptr;
		if (existingTexture != nullptr)
		{
			texture = existingTexture;
			texture->IncrementUpdateCount();
		}
		else
		{
			texture = Object::Create<Texture3D>();
		}
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