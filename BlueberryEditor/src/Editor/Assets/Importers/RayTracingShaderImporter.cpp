#include "RayTracingShaderImporter.h"

#include "Blueberry\Graphics\RayTracingShader.h"
#include "Blueberry\Graphics\GraphicsAPI.h"

#include "Editor\Assets\AssetDB.h"
#include "Editor\Assets\Processors\HLSLRayTracingShaderProcessor.h"
#include "Editor\Misc\PathHelper.h"
#include "ShaderImporter.h"

namespace Blueberry
{
	OBJECT_DEFINITION(RayTracingShaderImporter, AssetImporter)
	{
		DEFINE_BASE_FIELDS(RayTracingShaderImporter, AssetImporter)
	}

	String RayTracingShaderImporter::GetShaderFolder(const Guid& guid)
	{
		if (GraphicsAPI::GetAPI() == GraphicsAPI::API::DX12)
		{
			std::filesystem::path dataPath = Path::GetShaderCachePath();
			dataPath.append("DX12");
			dataPath.append(guid.ToString());
			if (!std::filesystem::exists(dataPath))
			{
				std::filesystem::create_directories(dataPath);
			}
			return StringHelper::ToString(dataPath);
		}
		return "";
	}

	bool RayTracingShaderImporter::IsRequiringReimport() const
	{
		Guid guid = GetGuid();
		if (AssetDB::HasAssetWithGuidInData(guid) && ShaderImporter::GetLastFilesWriteTime() < PathHelper::GetDirectoryLastWriteTime(GetShaderFolder(guid)))
		{
			return false;
		}
		return true;
	}

	void RayTracingShaderImporter::ImportData()
	{
		Guid guid = GetGuid();
		String path = GetFilePath();
		HLSLRayTracingShaderProcessor processor;

		if (processor.Compile(path))
		{
			processor.Save(GetShaderFolder(guid));
			RayTracingShader* rayTracingShader = GetOrCreateAssetObject<RayTracingShader>(1);
			rayTracingShader->SetName(GetName());
			rayTracingShader->Initialize(processor.GetShader(), processor.GetRayTracingShaderData());
			AssetDB::SaveAssetObjectsToCache(List<Object*> { rayTracingShader });
		}
		else
		{
			BB_ERROR("Ray tracing shader \"" << GetName() << "\" failed to compile.");
		}
		SetMainObject(1);
	}
}