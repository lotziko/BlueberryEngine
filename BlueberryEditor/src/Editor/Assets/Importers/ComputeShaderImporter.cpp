#include "ComputeShaderImporter.h"

#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Graphics\ComputeShader.h"

#include "Editor\Assets\AssetDB.h"
#include "Editor\Assets\Processors\HLSLComputeShaderProcessor.h"

namespace Blueberry
{
	OBJECT_DEFINITION(ComputeShaderImporter, AssetImporter)
	{
		DEFINE_BASE_FIELDS(ComputeShaderImporter, AssetImporter)
	}

	void ComputeShaderImporter::ImportData()
	{
		Guid guid = GetGuid();

		ComputeShader* object;
		if (AssetDB::HasAssetWithGuidInData(guid))
		{
			HLSLComputeShaderProcessor processor;
			if (processor.LoadKernels(GetShaderFolder()))
			{
				auto objects = AssetDB::LoadAssetObjects(guid, ObjectDB::GetObjectsFromGuid(guid));
				if (objects.size() == 1 && objects[0].first->IsClassType(ComputeShader::Type))
				{
					object = static_cast<ComputeShader*>(objects[0].first);
					ObjectDB::AllocateIdToGuid(object, guid, objects[0].second);
					object->Initialize(processor.GetShaders());
					object->SetState(ObjectState::Default);
					BB_INFO("Compute shader \"" << GetName() << "\" imported from cache.");
				}
			}
		}
		else
		{
			String path = GetFilePath();
			HLSLComputeShaderProcessor processor;

			if (processor.Compile(path))
			{
				processor.SaveKernels(GetShaderFolder());

				object = ComputeShader::Create(processor.GetShaders(), processor.GetComputeShaderData(), static_cast<ComputeShader*>(ObjectDB::GetObjectFromGuid(guid, 1)));
				object->SetState(ObjectState::Default);
				ObjectDB::AllocateIdToGuid(object, guid, 1);
				AssetDB::SaveAssetObjectsToCache(List<Object*> { object });
				BB_INFO("Compute shader \"" << GetName() << "\" imported and compiled from: " + path);
			}
			else
			{
				BB_ERROR("Compute shader \"" << GetName() << "\" failed to compile.");
			}
		}
		object->SetName(GetName());
		SetMainObject(1);
	}

	String ComputeShaderImporter::GetShaderFolder()
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