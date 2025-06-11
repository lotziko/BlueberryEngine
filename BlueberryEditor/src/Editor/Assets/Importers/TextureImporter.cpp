#include "TextureImporter.h"

#include "Blueberry\Graphics\Texture2D.h"
#include "Blueberry\Graphics\TextureCube.h"
#include "Blueberry\Graphics\GfxTexture.h"
#include "Blueberry\Graphics\GfxRenderTexturePool.h"
#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\StandardMeshes.h"
#include "Blueberry\Graphics\Shader.h"
#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Tools\FileHelper.h"
#include "Blueberry\Assets\AssetLoader.h"
#include "Editor\Assets\AssetDB.h"
#include "Editor\Assets\Processors\PngTextureProcessor.h"

#include "Blueberry\Graphics\Concrete\DefaultRenderer.h"

namespace Blueberry
{
	OBJECT_DEFINITION(TextureImporter, AssetImporter)
	{
		DEFINE_BASE_FIELDS(TextureImporter, AssetImporter)
		DEFINE_FIELD(TextureImporter, m_GenerateMipmaps, BindingType::Bool, {})
		DEFINE_FIELD(TextureImporter, m_IsSRGB, BindingType::Bool, {})
		DEFINE_FIELD(TextureImporter, m_WrapMode, BindingType::Enum, FieldOptions().SetEnumHint("Repeat,Clamp"))
		DEFINE_FIELD(TextureImporter, m_FilterMode, BindingType::Enum, FieldOptions().SetEnumHint("Linear,Point"))
		DEFINE_FIELD(TextureImporter, m_TextureShape, BindingType::Enum, FieldOptions().SetEnumHint("Texture2D,Texture2DArray,TextureCube,Texture3D"))
		DEFINE_FIELD(TextureImporter, m_TextureType, BindingType::Enum, {})
		DEFINE_FIELD(TextureImporter, m_TextureFormat, BindingType::Enum, {})
	}

	static Material* s_EquirectangularToCubemapMaterial = nullptr;

	const TextureImporterType& TextureImporter::GetTextureType()
	{
		return m_TextureType;
	}

	void TextureImporter::SetTextureType(const TextureImporterType& type)
	{
		m_TextureType = type;
	}

	const TextureImporterFormat& TextureImporter::GetTextureFormat()
	{
		return m_TextureFormat;
	}

	void TextureImporter::SetTextureFormat(const TextureImporterFormat& format)
	{
		m_TextureFormat = format;
	}

	const bool& TextureImporter::GetGenerateMipMaps()
	{
		return m_GenerateMipmaps;
	}

	void TextureImporter::SetGenerateMipMaps(const bool& generate)
	{
		m_GenerateMipmaps = generate;
	}

	const bool& TextureImporter::IsSRGB()
	{
		return m_IsSRGB;
	}

	void TextureImporter::SetSRGB(const bool& srgb)
	{
		m_IsSRGB = srgb;
	}

	void TextureImporter::ImportData()
	{
		static size_t Texture2DId = 1;
		static size_t TextureCubeId = 3;

		size_t id = static_cast<size_t>(m_TextureShape) + 1;
		Guid guid = GetGuid();
		// TODO check if dirty too

		Texture* object;
		if (AssetDB::HasAssetWithGuidInData(guid))
		{
			auto objects = AssetDB::LoadAssetObjects(guid, ObjectDB::GetObjectsFromGuid(guid));
			if (objects.size() == 1)
			{
				object = static_cast<Texture*>(objects[0].first);
				if (object->IsClassType(Texture2D::Type))
				{
					Texture2D* texture = static_cast<Texture2D*>(objects[0].first);
					ObjectDB::AllocateIdToGuid(object, guid, objects[0].second);
					uint8_t* data;
					size_t length;
					FileHelper::Load(data, length, GetTexturePath());
					texture->SetData(data, length);
					texture->Apply();
					texture->SetState(ObjectState::Default);
					//BB_INFO("Texture2D \"" << GetName() << "\" imported from cache.");
				}
				else if (object->IsClassType(TextureCube::Type))
				{
					TextureCube* texture = static_cast<TextureCube*>(objects[0].first);
					ObjectDB::AllocateIdToGuid(object, guid, objects[0].second);
					uint8_t* data;
					size_t length;
					FileHelper::Load(data, length, GetTexturePath());
					texture->SetData(data, length);
					texture->Apply();
					texture->SetState(ObjectState::Default);
					//BB_INFO("TextureCube \"" << GetName() << "\" imported from cache.");
				}
			}
		}
		else
		{
			String path = GetFilePath();
			
			PngTextureProcessor processor;
			std::string extension = std::filesystem::path(path).extension().string();
			if (extension == ".dds")
			{
				processor.LoadDDS(path);
			}
			else if (extension == ".hdr")
			{
				processor.LoadHDR(path);
			}
			else
			{
				processor.Load(path, m_IsSRGB, m_GenerateMipmaps);
				if (m_TextureType == TextureImporterType::NormalMap)
				{
					processor.CompressNormals();
				}
				if (m_TextureShape == TextureImporterShape::Texture2D)
				{
					processor.Compress(GetFormat(), m_IsSRGB);
				}
			}
			PngTextureProperties properties = processor.GetProperties();

			const auto& objects = ObjectDB::GetObjectsFromGuid(guid);
			auto it = objects.find(id);
			if (m_TextureShape == TextureImporterShape::Texture2D)
			{
				Texture2D* texture;
				if (it != objects.end())
				{
					texture = Texture2D::Create(properties.width, properties.height, properties.mipCount, properties.format, m_WrapMode, m_FilterMode, static_cast<Texture2D*>(ObjectDB::GetObject(it->second)));
				}
				else
				{
					texture = Texture2D::Create(properties.width, properties.height, properties.mipCount, properties.format, m_WrapMode, m_FilterMode);
					ObjectDB::AllocateIdToGuid(texture, guid, Texture2DId);
				}
				texture->SetState(ObjectState::Default);
				texture->SetData(properties.data, properties.dataSize);
				texture->Apply();
				object = texture;

				AssetDB::SaveAssetObjectsToCache(List<Object*> { object });
				FileHelper::Save(properties.data, properties.dataSize, GetTexturePath());
			}
			else if (m_TextureShape == TextureImporterShape::TextureCube)
			{
				TextureFormat uncompressedFormat = properties.format;
				TextureFormat compressedFormat = (extension == ".hdr" ? TextureFormat::BC6H_UFloat : GetFormat());
				
				TextureCube* texture;
				uint32_t size = std::min(properties.width, properties.height);
				if (it != objects.end())
				{
					texture = TextureCube::Create(size, size, 1, compressedFormat, m_WrapMode, m_FilterMode, static_cast<TextureCube*>(ObjectDB::GetObject(it->second)));
				}
				else
				{
					texture = TextureCube::Create(size, size, 1, compressedFormat, m_WrapMode, m_FilterMode);
					ObjectDB::AllocateIdToGuid(texture, guid, TextureCubeId);
				}

				Texture2D* temporaryTexture = Texture2D::Create(properties.width, properties.height, 1, uncompressedFormat);
				temporaryTexture->SetData(properties.data, properties.dataSize);
				temporaryTexture->Apply();
				GfxTexture* temporaryTextureCube = GfxRenderTexturePool::Get(size, size, 1, 1, uncompressedFormat, TextureDimension::TextureCube, WrapMode::Clamp, FilterMode::Linear, true);
				uint32_t blockSize = static_cast<uint32_t>(properties.dataSize / (properties.width * properties.height));
				size_t dataSize = size * size * 6 * blockSize;

				PngTextureProcessor cubeProcessor;
				cubeProcessor.CreateCube(uncompressedFormat, size, size);

				if (s_EquirectangularToCubemapMaterial == nullptr)
				{
					s_EquirectangularToCubemapMaterial = Material::Create(static_cast<Shader*>(AssetLoader::Load("assets/shaders/EquirectangularToCubemap.shader")));
				}

				GfxDevice::SetViewCount(6);
				GfxDevice::SetRenderTarget(temporaryTextureCube);
				GfxDevice::SetViewport(0, 0, size, size);
				GfxDevice::SetGlobalTexture(TO_HASH("_EquirectangularTexture"), temporaryTexture->Get());
				GfxDevice::Draw(GfxDrawingOperation(StandardMeshes::GetFullscreen(), s_EquirectangularToCubemapMaterial));
				GfxDevice::SetRenderTarget(nullptr);
				GfxDevice::SetViewCount(1);
				GfxDevice::Read(temporaryTextureCube, cubeProcessor.GetData());

				cubeProcessor.Compress(compressedFormat, m_IsSRGB);
				properties = cubeProcessor.GetProperties();

				Object::Destroy(temporaryTexture);
				GfxRenderTexturePool::Release(temporaryTextureCube);

				uint8_t* textureData = BB_MALLOC_ARRAY(uint8_t, properties.dataSize);
				memcpy(textureData, properties.data, properties.dataSize);
				texture->SetData(textureData, properties.dataSize);
				texture->Apply();
				texture->SetState(ObjectState::Default);
				object = texture;

				AssetDB::SaveAssetObjectsToCache(List<Object*> { object });
				FileHelper::Save(properties.data, properties.dataSize, GetTexturePath());
			}

			BB_INFO("Texture \"" << GetName() << "\" imported and created from: " + path);
		}
		object->SetName(GetName());
		SetMainObject(id);
	}

	String TextureImporter::GetTexturePath()
	{
		std::filesystem::path dataPath = Path::GetTextureCachePath();
		if (!std::filesystem::exists(dataPath))
		{
			std::filesystem::create_directories(dataPath);
		}
		dataPath.append(GetGuid().ToString().append(".texture"));
		return String(dataPath.string());
	}

	TextureFormat TextureImporter::GetFormat()
	{
		switch (m_TextureFormat)
		{
		case TextureImporterFormat::RGBA32:
			return m_IsSRGB ? TextureFormat::R8G8B8A8_UNorm_SRGB : TextureFormat::R8G8B8A8_UNorm;
		case TextureImporterFormat::RGB24:
			return m_IsSRGB ? TextureFormat::R8G8B8A8_UNorm_SRGB : TextureFormat::R8G8B8A8_UNorm;
		case TextureImporterFormat::RG16:
			return TextureFormat::R8G8_UNorm;
		case TextureImporterFormat::R8:
			return TextureFormat::R8_UNorm;
		case TextureImporterFormat::BC1:
			return m_IsSRGB ? TextureFormat::BC1_UNorm_SRGB : TextureFormat::BC1_UNorm;
		case TextureImporterFormat::BC3:
			return m_IsSRGB ? TextureFormat::BC3_UNorm_SRGB : TextureFormat::BC3_UNorm;
		case TextureImporterFormat::BC4:
			return TextureFormat::BC4_UNorm;
		case TextureImporterFormat::BC5:
			return TextureFormat::BC5_UNorm;
		case TextureImporterFormat::BC6H:
			return TextureFormat::BC6H_UFloat;
		case TextureImporterFormat::BC7:
			return m_IsSRGB ? TextureFormat::BC7_UNorm_SRGB : TextureFormat::BC7_UNorm;
		default:
			return TextureFormat::R8G8B8A8_UNorm;
		}
	}
}
