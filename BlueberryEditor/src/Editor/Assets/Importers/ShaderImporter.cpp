#include "ShaderImporter.h"

#include "Blueberry\Graphics\Shader.h"
#include "Editor\Assets\AssetDB.h"
#include "Editor\Assets\Processors\HLSLShaderParser.h"
#include "Editor\Assets\Processors\HLSLShaderProcessor.h"
#include "Editor\Misc\PathHelper.h"

namespace Blueberry
{
	OBJECT_DEFINITION(ShaderImporter, AssetImporter)
	{
		DEFINE_BASE_FIELDS(ShaderImporter, AssetImporter)
	}

	static long long s_LastFilesWriteTime = 0;

	String ShaderImporter::GetShaderFolder(const Guid& guid)
	{
		std::filesystem::path dataPath = Path::GetShaderCachePath();
		dataPath.append(guid.ToString());
		if (!std::filesystem::exists(dataPath))
		{
			std::filesystem::create_directories(dataPath);
		}
		return dataPath.string().data();
	}

	long long ShaderImporter::GetLastFilesWriteTime()
	{
		if (s_LastFilesWriteTime == 0)
		{
			String filesPath = "assets/shaders/";
			for (const auto& entry : std::filesystem::directory_iterator(filesPath))
			{
				std::filesystem::path path = entry;
				if (path.extension() == ".hlsl")
				{
					s_LastFilesWriteTime = std::max(s_LastFilesWriteTime, PathHelper::GetLastWriteTime(path));
				}
			}
		}
		return s_LastFilesWriteTime;
	}

	void ShaderImporter::ImportData()
	{
		Guid guid = GetGuid();

		Shader* object;
		String folderPath = GetShaderFolder(guid);
		if (AssetDB::HasAssetWithGuidInData(guid) && GetLastFilesWriteTime() < PathHelper::GetDirectoryLastWriteTime(folderPath))
		{
			HLSLShaderProcessor processor;
			if (processor.LoadVariants(folderPath))
			{
				auto objects = AssetDB::LoadAssetObjects(guid, ObjectDB::GetObjectsFromGuid(guid));
				if (objects.size() == 1 && objects[0].first->IsClassType(Shader::Type))
				{
					object = static_cast<Shader*>(objects[0].first);
					ObjectDB::AllocateIdToGuid(object, guid, objects[0].second);
					object->Initialize(processor.GetVariantsData());
					object->SetState(ObjectState::Default);
					BB_INFO("Shader \"" << GetName() << "\" imported from cache.");
				}
			}
		}
		else
		{
			String path = GetFilePath();
			HLSLShaderProcessor processor;
			
			if (processor.Compile(path))
			{
				processor.SaveVariants(folderPath);

				object = Shader::Create(processor.GetVariantsData(), processor.GetShaderData(), static_cast<Shader*>(ObjectDB::GetObjectFromGuid(guid, 1)));
				object->SetState(ObjectState::Default);
				ObjectDB::AllocateIdToGuid(object, guid, 1);
				AssetDB::SaveAssetObjectsToCache(List<Object*> { object });
				BB_INFO("Shader \"" << GetName() << "\" imported and compiled from: " + path);
			}
			else
			{
				BB_ERROR("Shader \"" << GetName() << "\" failed to compile.");
			}
		}
		object->SetName(GetName());
		SetMainObject(1);
	}
}
