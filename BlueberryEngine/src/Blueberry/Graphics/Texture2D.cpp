#include "bbpch.h"
#include "Texture2D.h"

#include <iomanip>
#include "Blueberry\Core\GlobalServices.h"
#include "Blueberry\Tools\ByteConverter.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Texture, Texture2D)
	
	Texture2D::Texture2D(const TextureProperties& properties)
	{
		g_GraphicsDevice->CreateTexture(properties, m_Texture);
		m_RawData = reinterpret_cast<BYTE*>(properties.data);
		m_RawDataSize = properties.dataSize;
	}

	void Texture2D::Serialize(SerializationContext& context, ryml::NodeRef& node)
	{
		size_t length = m_RawDataSize * 2;
		char* dst = new char[length];
		ByteConverter::BytesToHexString(m_RawData, dst, m_RawDataSize);

		node["m_Width"] << m_Texture->GetWidth();
		node["m_Height"] << m_Texture->GetHeight();
		node["m_RawDataSize"] << m_RawDataSize;
		node["m_RawData"] << ryml::substr(dst, length);

		delete[] dst;
	}

	void Texture2D::Deserialize(SerializationContext& context, ryml::NodeRef& node)
	{
		UINT width, height;
		std::string dataString;

		node["m_Width"] >> width;
		node["m_Height"] >> height;
		node["m_RawData"] >> dataString; // TODO think if it's possible to do it faster without string
		node["m_RawDataSize"] >> m_RawDataSize;
		
		BYTE* data = new BYTE[m_RawDataSize];

		ByteConverter::HexStringToBytes(dataString.c_str(), data, m_RawDataSize);

		TextureProperties properties;
		properties.width = width;
		properties.height = height;
		properties.dataSize = m_RawDataSize;
		properties.data = data;
		properties.isRenderTarget = false;

		g_GraphicsDevice->CreateTexture(properties, m_Texture);
		m_RawData = data;
		m_RawDataSize = properties.dataSize;
	}

	Ref<Texture2D> Texture2D::Create(const TextureProperties& properties)
	{
		return ObjectDB::CreateObject<Texture2D>(properties);
	}
}