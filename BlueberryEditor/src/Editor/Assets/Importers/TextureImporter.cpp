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
		DEFINE_FIELD(TextureImporter, m_GenerateMipmaps, BindingType::Bool, FieldOptions())
		DEFINE_FIELD(TextureImporter, m_IsSRGB, BindingType::Bool, FieldOptions())
		DEFINE_FIELD(TextureImporter, m_IsReadable, BindingType::Bool, FieldOptions())
		DEFINE_FIELD(TextureImporter, m_WrapMode, BindingType::Enum, FieldOptions().SetEnumHint("Repeat,Clamp"))
		DEFINE_FIELD(TextureImporter, m_FilterMode, BindingType::Enum, FieldOptions().SetEnumHint("Point,Bilinear,Trilinear,Anisotropic"))
		DEFINE_FIELD(TextureImporter, m_TextureShape, BindingType::Enum, FieldOptions().SetEnumHint("Texture2D,Texture2DArray,TextureCube,Texture3D"))
		DEFINE_FIELD(TextureImporter, m_TextureType, BindingType::Enum, FieldOptions().SetEnumHint("Default,BaseMap,NormalMap,Mask,Cookie"))
		DEFINE_FIELD(TextureImporter, m_TextureFormat, BindingType::Enum, FieldOptions())
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

	const bool& TextureImporter::IsReadable()
	{
		return m_IsReadable;
	}

	void TextureImporter::SetReadable(const bool& readable)
	{
		m_IsReadable = readable;
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

	bool TextureImporter::IsRequiringReimport() const
	{
		Guid guid = GetGuid();
		FileId id = static_cast<size_t>(m_TextureShape) + 1;
		if (AssetDB::HasAssetWithGuidInData(guid) && ObjectDB::HasGuidAndFileId(guid, id))
		{
			return false;
		}
		return true;
	}

	void TextureImporter::ImportData()
	{
		Guid guid = GetGuid();
		FileId id = static_cast<size_t>(m_TextureShape) + 1;
		String texturePath = GetTexturePath(guid);

		String path = GetFilePath();
		String extension = String(std::filesystem::path(path).extension().string());
		DirectX::ScratchImage image = {};
		TextureHelper::Load(image, path, extension, m_IsSRGB);

		if (image.GetPixelsSize() == 0)
		{
			BB_ERROR("Failed to load texture.");
			return;
		}

		if (m_TextureShape == TextureShape::Texture2D)
		{
			Blueberry::TextureFormat format;
			if (extension != ".dds")
			{
				if (m_TextureType == TextureType::NormalMap)
				{
					TextureHelper::CompressNormals(image);
				}
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
			auto& metadata = image.GetMetadata();

			Texture2D* texture = GetOrCreateAssetObject<Texture2D>(id);
			texture->SetName(GetName());
			texture->Initialize(static_cast<uint32_t>(metadata.width), static_cast<uint32_t>(metadata.height), static_cast<uint32_t>(metadata.mipLevels), format);
			texture->SetWrapMode(m_WrapMode);
			texture->SetFilterMode(m_FilterMode);
			texture->SetReadable(m_IsReadable);
			texture->SetData(image.GetPixels(), image.GetPixelsSize());
			texture->Apply();

			AssetDB::SaveAssetObjectsToCache(List<Object*> { texture });
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

			auto& metadata = image.GetMetadata();
			uint32_t size = static_cast<uint32_t>(std::min(metadata.width, metadata.height));
			TextureCube* texture = GetOrCreateAssetObject<TextureCube>(id);
			texture->SetName(GetName());
			texture->Initialize(size, size, static_cast<uint32_t>(metadata.mipLevels), compressedFormat);
			texture->SetWrapMode(m_WrapMode);
			texture->SetFilterMode(m_FilterMode);
			texture->SetReadable(m_IsReadable);
			texture->SetData(image.GetPixels(), image.GetPixelsSize());
			texture->Apply();

			AssetDB::SaveAssetObjectsToCache(List<Object*> { texture });
			FileHelper::Save(image.GetPixels(), image.GetPixelsSize(), texturePath);
		}
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
