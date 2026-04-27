#include "ComputeShaderImporter.h"

#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Graphics\ComputeShader.h"

#include "Editor\Assets\AssetDB.h"
#include "Editor\Assets\Processors\HLSLComputeShaderProcessor.h"
#include "Editor\Misc\PathHelper.h"
#include "ShaderImporter.h"

namespace Blueberry
{
	OBJECT_DEFINITION(ComputeShaderImporter, AssetImporter)
	{
		DEFINE_BASE_FIELDS(ComputeShaderImporter, AssetImporter)
	}

	String ComputeShaderImporter::GetShaderFolder(const Guid& guid)
	{
		std::filesystem::path dataPath = Path::GetShaderCachePath();
		dataPath.append(guid.ToString());
		if (!std::filesystem::exists(dataPath))
		{
			std::filesystem::create_directories(dataPath);
		}
		return StringHelper::ToString(dataPath);
	}

	bool ComputeShaderImporter::IsRequiringReimport() const
	{
		Guid guid = GetGuid();
		if (AssetDB::HasAssetWithGuidInData(guid) && ShaderImporter::GetLastFilesWriteTime() < PathHelper::GetDirectoryLastWriteTime(GetShaderFolder(guid)))
		{
			return false;
		}
		return true;
	}

	void ComputeShaderImporter::ImportData()
	{
		Guid guid = GetGuid();
		String path = GetFilePath();
		HLSLComputeShaderProcessor processor;

		if (processor.Compile(path))
		{
			processor.SaveKernels(GetShaderFolder(guid));
			ComputeShader* computeShader = GetOrCreateAssetObject<ComputeShader>(1);
			computeShader->SetName(GetName());
			computeShader->Initialize(processor.GetShaders(), processor.GetComputeShaderData());
			AssetDB::SaveAssetObjectsToCache(List<Object*> { computeShader });
		}
		else
		{
			BB_ERROR("Compute shader \"" << GetName() << "\" failed to compile.");
		}
		SetMainObject(1);
	}
}