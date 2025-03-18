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
	OBJECT_DEFINITION(AssetImporter, TextureImporter)

	static Material* s_EquirectangularToCubemapMaterial = nullptr;

	void TextureImporter::BindProperties()
	{
		BEGIN_OBJECT_BINDING(TextureImporter)
		BIND_FIELD(FieldInfo(TO_STRING(m_GenerateMipmaps), &TextureImporter::m_GenerateMipmaps, BindingType::Bool))
		BIND_FIELD(FieldInfo(TO_STRING(m_IsSRGB), &TextureImporter::m_IsSRGB, BindingType::Bool))
		BIND_FIELD(FieldInfo(TO_STRING(m_WrapMode), &TextureImporter::m_WrapMode, BindingType::Enum).SetHintData("Repeat,Clamp"))
		BIND_FIELD(FieldInfo(TO_STRING(m_FilterMode), &TextureImporter::m_FilterMode, BindingType::Enum).SetHintData("Linear,Point"))
		BIND_FIELD(FieldInfo(TO_STRING(m_TextureShape), &TextureImporter::m_TextureShape, BindingType::Enum).SetHintData("Texture2D,Texture2DArray,TextureCube,Texture3D"))
		END_OBJECT_BINDING()
	}

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
			if (std::filesystem::path(path).extension() == ".dds")
			{
				processor.LoadDDS(path);
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
				texture->SetData(static_cast<uint8_t*>(properties.data), properties.dataSize);
				texture->Apply();
				object = texture;

				AssetDB::SaveAssetObjectsToCache(List<Object*> { object });
				FileHelper::Save(static_cast<uint8_t*>(properties.data), properties.dataSize, GetTexturePath());
			}
			else if (m_TextureShape == TextureImporterShape::TextureCube)
			{
				TextureCube* texture;
				uint32_t size = std::min(properties.width, properties.height);
				if (it != objects.end())
				{
					texture = TextureCube::Create(size, size, properties.mipCount, properties.format, m_WrapMode, m_FilterMode, static_cast<TextureCube*>(ObjectDB::GetObject(it->second)));
				}
				else
				{
					texture = TextureCube::Create(size, size, properties.mipCount, properties.format, m_WrapMode, m_FilterMode);
					ObjectDB::AllocateIdToGuid(texture, guid, TextureCubeId);
				}

				Texture2D* temporaryTexture = Texture2D::Create(properties.width, properties.height, 1, TextureFormat::R8G8B8A8_UNorm);
				temporaryTexture->SetData(static_cast<uint8_t*>(properties.data), properties.dataSize);
				temporaryTexture->Apply();
				RenderTexture* temporaryTextureCube = RenderTexture::Create(size, size, 1, 1, TextureFormat::R8G8B8A8_UNorm, TextureDimension::TextureCube, WrapMode::Clamp, FilterMode::Linear, true);
				size_t dataSize = size * size * 6 * 4;
				uint8_t* data = BB_MALLOC_ARRAY(uint8_t, dataSize);

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
				GfxDevice::Read(temporaryTextureCube->Get(), data);

				Object::Destroy(temporaryTexture);
				Object::Destroy(temporaryTextureCube);

				texture->SetData(data, dataSize);
				texture->Apply();
				texture->SetState(ObjectState::Default);
				object = texture;

				AssetDB::SaveAssetObjectsToCache(List<Object*> { object });
				FileHelper::Save(data, dataSize, GetTexturePath());
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
