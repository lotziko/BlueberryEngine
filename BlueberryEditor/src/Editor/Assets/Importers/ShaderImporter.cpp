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

	bool ShaderImporter::IsRequiringReimport() const
	{
		Guid guid = GetGuid();
		if (AssetDB::HasAssetWithGuidInData(guid) && GetLastFilesWriteTime() < PathHelper::GetDirectoryLastWriteTime(GetShaderFolder(guid)))
		{
			return false;
		}
		return true;
	}

	void ShaderImporter::ImportData()
	{
		Guid guid = GetGuid();
		String folderPath = GetShaderFolder(guid);
		String path = GetFilePath();
		HLSLShaderProcessor processor;

		if (processor.Compile(path))
		{
			processor.SaveVariants(folderPath);
			Shader* shader = GetOrCreateAssetObject<Shader>(1);
			shader->SetName(GetName());
			shader->Initialize(processor.GetVariantsData(), processor.GetShaderData());
			AssetDB::SaveAssetObjectsToCache(List<Object*> { shader });
		}
		else
		{
			BB_ERROR("Shader \"" << GetName() << "\" failed to compile.");
		}
		SetMainObject(1);
	}
}
