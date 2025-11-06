#include "TextureImporter.h"

#include "Blueberry\Graphics\Texture2D.h"
#include "Blueberry\Graphics\TextureCube.h"
#include "Blueberry\Tools\FileHelper.h"
#include "Editor\Assets\AssetDB.h"
#include "Editor\Misc\TextureHelper.h"
#include "Editor\Assets\Processors\ReflectionGenerator.h"

#include <directxtex\DirectXTex.h>

namespace Blueberry
{
	OBJECT_DEFINITION(TextureImporter, AssetImporter)
	{
		DEFINE_BASE_FIELDS(TextureImporter, AssetImporter)
		DEFINE_FIELD(TextureImporter, m_GenerateMipmaps, BindingType::Bool, {})
		DEFINE_FIELD(TextureImporter, m_IsSRGB, BindingType::Bool, {})
		DEFINE_FIELD(TextureImporter, m_WrapMode, BindingType::Enum, FieldOptions().SetEnumHint("Repeat,Clamp"))
		DEFINE_FIELD(TextureImporter, m_FilterMode, BindingType::Enum, FieldOptions().SetEnumHint("Point,Bilinear,Trilinear,Anisotropic"))
		DEFINE_FIELD(TextureImporter, m_TextureShape, BindingType::Enum, FieldOptions().SetEnumHint("Texture2D,Texture2DArray,TextureCube,Texture3D"))
		DEFINE_FIELD(TextureImporter, m_TextureType, BindingType::Enum, FieldOptions().SetEnumHint("Default,BaseMap,NormalMap,Mask,Cookie"))
		DEFINE_FIELD(TextureImporter, m_TextureFormat, BindingType::Enum, {})
		DEFINE_FIELD(TextureImporter, m_TextureCubeType, BindingType::Enum, FieldOptions().SetEnumHint("Equirectangular,Slices"))
		DEFINE_FIELD(TextureImporter, m_TextureCubeIBLType, BindingType::Enum, FieldOptions().SetEnumHint("None,Specular"))
	}

	const TextureImporter::TextureShape& TextureImporter::GetTextureShape()
	{
		return m_TextureShape;
	}

	void TextureImporter::SetTextureShape(const TextureShape& shape)
	{
		m_TextureShape = shape;
	}

	const TextureImporter::TextureType& TextureImporter::GetTextureType()
	{
		return m_TextureType;
	}

	void TextureImporter::SetTextureType(const TextureType& type)
	{
		m_TextureType = type;
	}

	const TextureImporter::TextureFormat& TextureImporter::GetTextureFormat()
	{
		return m_TextureFormat;
	}

	void TextureImporter::SetTextureFormat(const TextureFormat& format)
	{
		m_TextureFormat = format;
	}

	const TextureImporter::TextureCubeType& TextureImporter::GetTextureCubeType()
	{
		return m_TextureCubeType;
	}

	void TextureImporter::SetTextureCubeType(const TextureCubeType& cubeType)
	{
		m_TextureCubeType = cubeType;
	}

	const TextureImporter::TextureCubeIBLType& TextureImporter::GetTextureCubeIBLType()
	{
		return m_TextureCubeIBLType;
	}

	void TextureImporter::SetTextureCubeIBLType(const TextureCubeIBLType& cubeIBLType)
	{
		m_TextureCubeIBLType = cubeIBLType;
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

	const WrapMode& TextureImporter::GetWrapMode()
	{
		return m_WrapMode;
	}

	void TextureImporter::SetWrapMode(const WrapMode& wrapMode)
	{
		m_WrapMode = wrapMode;
	}

	const FilterMode& TextureImporter::GetFilterMode()
	{
		return m_FilterMode;
	}

	void TextureImporter::SetFilterMode(const FilterMode& filterMode)
	{
		m_FilterMode = filterMode;
	}

	String TextureImporter::GetTexturePath(const Guid& guid)
	{
		std::filesystem::path dataPath = Path::GetTextureCachePath();
		if (!std::filesystem::exists(dataPath))
		{
			std::filesystem::create_directories(dataPath);
		}
		dataPath.append(guid.ToString().append(".texture"));
		return String(dataPath.string());
	}

	void TextureImporter::ImportData()
	{
		Guid guid = GetGuid();
		FileId id = static_cast<size_t>(m_TextureShape) + 1;
		String texturePath = GetTexturePath(guid);

		Texture* object;
		if (AssetDB::HasAssetWithGuidInData(guid) && ObjectDB::HasGuidAndFileId(guid, id))
		{
			// TODO what if texture in cache does not exist
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
					FileHelper::Load(data, length, texturePath);
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
					FileHelper::Load(data, length, texturePath);
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
			String extension = String(std::filesystem::path(path).extension().string());
			DirectX::ScratchImage image = {};
			TextureHelper::Load(image, path, extension, m_IsSRGB);
			TextureHelper::Flip(image);
			
			const auto& objects = ObjectDB::GetObjectsFromGuid(guid);
			auto it = objects.find(id);
			if (m_TextureShape == TextureShape::Texture2D)
			{
				Blueberry::TextureFormat format;
				if (extension != ".dds")
				{
					if (m_GenerateMipmaps)
					{
						TextureHelper::GenerateMipMaps(image);
					}
					format = GetFormat();
					TextureHelper::Compress(image, format, m_IsSRGB);
				}
				else
				{
					format = static_cast<Blueberry::TextureFormat>(image.GetMetadata().format);
				}
				auto metadata = image.GetMetadata();

				Texture2D* texture;
				if (it != objects.end())
				{
					texture = Texture2D::Create(metadata.width, metadata.height, metadata.mipLevels, format, m_WrapMode, m_FilterMode, static_cast<Texture2D*>(ObjectDB::GetObject(it->second)));
				}
				else
				{
					texture = Texture2D::Create(metadata.width, metadata.height, metadata.mipLevels, format, m_WrapMode, m_FilterMode);
					ObjectDB::AllocateIdToGuid(texture, guid, id);
				}

				uint8_t* textureData = BB_MALLOC_ARRAY(uint8_t, image.GetPixelsSize());
				memcpy(textureData, image.GetPixels(), image.GetPixelsSize());
				texture->SetState(ObjectState::Default);
				texture->SetData(textureData, image.GetPixelsSize());
				texture->Apply();
				object = texture;

				AssetDB::SaveAssetObjectsToCache(List<Object*> { object });
				FileHelper::Save(image.GetPixels(), image.GetPixelsSize(), texturePath);
			}
			else if (m_TextureShape == TextureShape::TextureCube)
			{
				Blueberry::TextureFormat uncompressedFormat = static_cast<Blueberry::TextureFormat>(image.GetMetadata().format);
				Blueberry::TextureFormat compressedFormat = (extension == ".hdr" ? Blueberry::TextureFormat::BC6H_UFloat : GetFormat());

				if (m_TextureCubeType == TextureCubeType::Equirectangular)
				{
					TextureHelper::EquirectangularToTextureCube(image, uncompressedFormat);
				}
				else if (m_TextureCubeType == TextureCubeType::Slices)
				{
					TextureHelper::SlicesToTextureCube(image);
				}

				if (m_TextureCubeIBLType == TextureCubeIBLType::Specular)
				{
					TextureHelper::ConvoluteSpecularTextureCube(image);
				}

				TextureHelper::Compress(image, compressedFormat, m_IsSRGB);

				auto metadata = image.GetMetadata();
				TextureCube* texture;
				uint32_t size = std::min(metadata.width, metadata.height);
				uint32_t mipCount = metadata.mipLevels;
				if (it != objects.end())
				{
					texture = TextureCube::Create(size, size, mipCount, compressedFormat, m_WrapMode, m_FilterMode, static_cast<TextureCube*>(ObjectDB::GetObject(it->second)));
				}
				else
				{
					texture = TextureCube::Create(size, size, mipCount, compressedFormat, m_WrapMode, m_FilterMode);
					ObjectDB::AllocateIdToGuid(texture, guid, id);
				}

				uint8_t* textureData = BB_MALLOC_ARRAY(uint8_t, image.GetPixelsSize());
				memcpy(textureData, image.GetPixels(), image.GetPixelsSize());
				texture->SetData(textureData, image.GetPixelsSize());
				texture->Apply();
				texture->SetState(ObjectState::Default);
				object = texture;

				AssetDB::SaveAssetObjectsToCache(List<Object*> { object });
				FileHelper::Save(image.GetPixels(), image.GetPixelsSize(), texturePath);
			}
			//BB_INFO("Texture \"" << GetName() << "\" imported and created from: " + path);
		}
		object->SetName(GetName());
		SetMainObject(id);
	}

	Blueberry::TextureFormat TextureImporter::GetFormat()
	{
		switch (m_TextureFormat)
		{
		case TextureFormat::RGBA32:
			return m_IsSRGB ? Blueberry::TextureFormat::R8G8B8A8_UNorm_SRGB : Blueberry::TextureFormat::R8G8B8A8_UNorm;
		case TextureFormat::RGB24:
			return m_IsSRGB ? Blueberry::TextureFormat::R8G8B8A8_UNorm_SRGB : Blueberry::TextureFormat::R8G8B8A8_UNorm;
		case TextureFormat::RG16:
			return Blueberry::TextureFormat::R8G8_UNorm;
		case TextureFormat::R8:
			return Blueberry::TextureFormat::R8_UNorm;
		case TextureFormat::BC1:
			return m_IsSRGB ? Blueberry::TextureFormat::BC1_UNorm_SRGB : Blueberry::TextureFormat::BC1_UNorm;
		case TextureFormat::BC3:
			return m_IsSRGB ? Blueberry::TextureFormat::BC3_UNorm_SRGB : Blueberry::TextureFormat::BC3_UNorm;
		case TextureFormat::BC4:
			return Blueberry::TextureFormat::BC4_UNorm;
		case TextureFormat::BC5:
			return Blueberry::TextureFormat::BC5_UNorm;
		case TextureFormat::BC6H:
			return Blueberry::TextureFormat::BC6H_UFloat;
		case TextureFormat::BC7:
			return m_IsSRGB ? Blueberry::TextureFormat::BC7_UNorm_SRGB : Blueberry::TextureFormat::BC7_UNorm;
		default:
			return Blueberry::TextureFormat::R8G8B8A8_UNorm;
		}
	}
}
