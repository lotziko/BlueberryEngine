#include "ShaderImporter.h"

#include "Blueberry\Graphics\Shader.h"
#include "Editor\Assets\AssetDB.h"
#include "Editor\Assets\Processors\HLSLShaderParser.h"
#include "Editor\Assets\Processors\HLSLShaderProcessor.h"

namespace Blueberry
{
	OBJECT_DEFINITION(ShaderImporter, AssetImporter)
	{
		DEFINE_BASE_FIELDS(ShaderImporter, AssetImporter)
	}

	void ShaderImporter::ImportData()
	{
		Guid guid = GetGuid();

		Shader* object;
		if (AssetDB::HasAssetWithGuidInData(guid))
		{
			HLSLShaderProcessor processor;
			if (processor.LoadVariants(GetShaderFolder()))
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
				processor.SaveVariants(GetShaderFolder());

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

	String ShaderImporter::GetShaderFolder()
	{
		std::filesystem::path dataPath = Path::GetShaderCachePath();
		dataPath.append(GetGuid().ToString());
		if (!std::filesystem::exists(dataPath))
		{
			std::filesystem::create_directories(dataPath);
		}
		return dataPath.string().data();
	}
}
