#include "bbpch.h"
#include "TextureImporter.h"

#include "Blueberry\Graphics\Texture2D.h"
#include "Blueberry\Graphics\TextureCube.h"
#include "Blueberry\Graphics\RenderTexture.h"
#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\StandardMeshes.h"
#include "Blueberry\Graphics\Shader.h"
#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Tools\FileHelper.h"
#include "Blueberry\Assets\AssetLoader.h"
#include "Editor\Assets\AssetDB.h"
#include "Editor\Assets\Processors\PngTextureProcessor.h"

#include "Blueberry\Graphics\DefaultRenderer.h"

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
	}

	static Material* s_EquirectangularToCubemapMaterial = nullptr;

	void TextureImporter::ImportData()
	{
		static std::size_t Texture2DId = 1;
		static std::size_t TextureCubeId = 3;

		std::size_t id = static_cast<std::size_t>(m_TextureShape) + 1;
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
			std::string path = GetFilePath();
			
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
				if (m_TextureShape == TextureImporterShape::Texture2D)
				{
					processor.Compress(TextureFormat::BC7_UNorm);
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
				TextureFormat originalFormat = properties.format;
				TextureFormat compressedFormat = TextureFormat::BC6H_UFloat;

				TextureCube* texture;
				uint32_t size = std::min(properties.width, properties.height);
				if (it != objects.end())
				{
					texture = TextureCube::Create(size, size, properties.mipCount, compressedFormat, m_WrapMode, m_FilterMode, static_cast<TextureCube*>(ObjectDB::GetObject(it->second)));
				}
				else
				{
					texture = TextureCube::Create(size, size, properties.mipCount, compressedFormat, m_WrapMode, m_FilterMode);
					ObjectDB::AllocateIdToGuid(texture, guid, TextureCubeId);
				}

				Texture2D* temporaryTexture = Texture2D::Create(properties.width, properties.height, 1, originalFormat);
				temporaryTexture->SetData(properties.data, properties.dataSize);
				temporaryTexture->Apply();
				RenderTexture* temporaryTextureCube = RenderTexture::Create(size, size, 1, 1, originalFormat, TextureDimension::TextureCube, WrapMode::Clamp, FilterMode::Linear, true);
				uint8_t blockSize = properties.dataSize / (properties.width * properties.height);
				size_t dataSize = size * size * 6 * blockSize;

				PngTextureProcessor cubeProcessor;
				cubeProcessor.CreateCube(originalFormat, size, size);

				if (s_EquirectangularToCubemapMaterial == nullptr)
				{
					s_EquirectangularToCubemapMaterial = Material::Create(static_cast<Shader*>(AssetLoader::Load("assets/shaders/EquirectangularToCubemap.shader")));
				}

				GfxDevice::SetViewCount(6);
				GfxDevice::SetRenderTarget(temporaryTextureCube->Get());
				GfxDevice::SetViewport(0, 0, size, size);
				GfxDevice::SetGlobalTexture(TO_HASH("_BaseMap"), temporaryTexture->Get());
				GfxDevice::Draw(GfxDrawingOperation(StandardMeshes::GetFullscreen(), s_EquirectangularToCubemapMaterial));
				GfxDevice::SetRenderTarget(nullptr);
				GfxDevice::SetViewCount(1);
				GfxDevice::Read(temporaryTextureCube->Get(), cubeProcessor.GetData());

				cubeProcessor.Compress(compressedFormat);
				properties = cubeProcessor.GetProperties();

				Object::Destroy(temporaryTexture);
				Object::Destroy(temporaryTextureCube);

				texture->SetData(properties.data, properties.dataSize);
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

	std::string TextureImporter::GetTexturePath()
	{
		std::filesystem::path dataPath = Path::GetTextureCachePath();
		if (!std::filesystem::exists(dataPath))
		{
			std::filesystem::create_directories(dataPath);
		}
		dataPath.append(GetGuid().ToString().append(".texture"));
		return dataPath.string();
	}
}
