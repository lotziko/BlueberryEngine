#include "bbpch.h"
#include "ShaderImporter.h"

#include "Blueberry\Graphics\Shader.h"
#include "Editor\Assets\AssetDB.h"
#include "Editor\ShaderProcessor.h"

namespace Blueberry
{
	OBJECT_DEFINITION(AssetImporter, ShaderImporter)

	void ShaderImporter::BindProperties()
	{
		
	}

	void ShaderImporter::ImportData()
	{
		Guid guid = GetGuid();
		std::string vertexPath = GetShaderPath(".vertex");
		std::string fragmentPath = GetShaderPath(".fragment");

		Shader* object;
		if (AssetDB::HasAssetWithGuidInData(guid))
		{
			void* vertex = ShaderProcessor::Load(vertexPath);
			void* fragment = ShaderProcessor::Load(fragmentPath);
			object = AssetDB::LoadAssetObject<Shader>(guid);
			object->Initialize(vertex, fragment);
			BB_INFO(std::string() << "Shader \"" << GetName() << "\" imported from cache.");
		}
		else
		{
			std::string path = GetFilePath();
			void* vertex = ShaderProcessor::Compile(path, "Vertex", "vs_5_0", vertexPath);
			void* fragment = ShaderProcessor::Compile(path, "Fragment", "ps_5_0", fragmentPath);
			object = Shader::Create(vertex, fragment);
			ObjectDB::AllocateIdToGuid(object, guid);
			AssetDB::SaveAssetObjectToCache(object);
			BB_INFO(std::string() << "Shader \"" << GetName() << "\" imported and compiled from: " + path);
		}
		object->SetName(GetName());
		AddImportedObject(object);
	}

	std::string ShaderImporter::GetShaderPath(const char* extension)
	{
		std::filesystem::path dataPath = Path::GetShaderCachePath();
		dataPath.append(GetGuid().ToString());
		if (!std::filesystem::exists(dataPath))
		{
			std::filesystem::create_directories(dataPath);
		}
		return dataPath.append(std::string("0").append(extension)).string();
	}
}
