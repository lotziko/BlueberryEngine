#include "bbpch.h"
#include "TextureImporter.h"

#include "Blueberry\Graphics\Texture2D.h"
#include "Blueberry\Tools\FileHelper.h"
#include "Editor\Assets\AssetDB.h"
#include "Editor\Assets\Processors\PngTextureProcessor.h"

namespace Blueberry
{
	OBJECT_DEFINITION(AssetImporter, TextureImporter)

	void TextureImporter::BindProperties()
	{
		BEGIN_OBJECT_BINDING(TextureImporter)
		BIND_FIELD(FieldInfo(TO_STRING(m_GenerateMipmaps), &TextureImporter::m_GenerateMipmaps, BindingType::Bool))
		BIND_FIELD(FieldInfo(TO_STRING(m_IsSRGB), &TextureImporter::m_IsSRGB, BindingType::Bool))
		BIND_FIELD(FieldInfo(TO_STRING(m_WrapMode), &TextureImporter::m_WrapMode, BindingType::Enum).SetHintData("Repeat,Clamp"))
		BIND_FIELD(FieldInfo(TO_STRING(m_FilterMode), &TextureImporter::m_FilterMode, BindingType::Enum).SetHintData("Linear,Point"))
		END_OBJECT_BINDING()
	}

	void TextureImporter::ImportData()
	{
		static std::size_t TextureId = 1;

		Guid guid = GetGuid();
		// TODO check if dirty too

		Texture2D* object;
		if (AssetDB::HasAssetWithGuidInData(guid))
		{
			auto objects = AssetDB::LoadAssetObjects(guid, ObjectDB::GetObjectsFromGuid(guid));
			if (objects.size() == 1 && objects[0].first->IsClassType(Texture2D::Type))
			{
				object = static_cast<Texture2D*>(objects[0].first);
				ObjectDB::AllocateIdToGuid(object, guid, objects[0].second);
				byte* data;
				size_t length;
				FileHelper::Load(data, length, GetTexturePath());
				object->SetData(data, length);
				object->Apply();
				object->SetState(ObjectState::Default);
				delete[] data;
				BB_INFO("Texture \"" << GetName() << "\" imported from cache.");
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
				processor.Compress(TextureFormat::BC7_UNorm);
			}
			PngTextureProperties properties = processor.GetProperties();

			const auto& objects = ObjectDB::GetObjectsFromGuid(guid);
			auto it = objects.find(TextureId);
			if (it != objects.end())
			{
				object = Texture2D::Create(properties.width, properties.height, properties.mipCount, properties.format, m_WrapMode, m_FilterMode, (Texture2D*)ObjectDB::GetObject(it->second));
				object->SetState(ObjectState::Default);
			}
			else
			{
				object = Texture2D::Create(properties.width, properties.height, properties.mipCount, properties.format, m_WrapMode, m_FilterMode);
				ObjectDB::AllocateIdToGuid(object, guid, TextureId);
			}
			object->SetData((byte*)properties.data, properties.dataSize);
			object->Apply();

			AssetDB::SaveAssetObjectsToCache(std::vector<Object*> { object });
			FileHelper::Save((byte*)properties.data, properties.dataSize, GetTexturePath());

			BB_INFO("Texture \"" << GetName() << "\" imported and created from: " + path);
		}
		object->SetName(GetName());
		SetMainObject(TextureId);
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
