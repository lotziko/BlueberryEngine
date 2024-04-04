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
		if (ObjectDB::HasGuid(guid))
		{
			BB_INFO("Shader \"" << GetName() << "\" is already imported.");
		}
		if (AssetDB::HasAssetWithGuidInData(guid))
		{
			void* vertex = ShaderProcessor::Load(vertexPath);
			void* fragment = ShaderProcessor::Load(fragmentPath);
			std::vector<Object*> objects = AssetDB::LoadAssetObjects(guid);
			if (objects.size() == 1 && objects[0]->IsClassType(Shader::Type))
			{
				object = static_cast<Shader*>(objects[0]);
				object->Initialize(vertex, fragment);
				BB_INFO("Shader \"" << GetName() << "\" imported from cache.");
			}
		}
		else
		{
			std::string path = GetFilePath();
			std::string shaderData;
			RawShaderOptions options;
			ShaderProcessor::Process(path, shaderData, options);
			void* vertex = ShaderProcessor::Compile(shaderData, "Vertex", "vs_5_0", vertexPath);
			void* fragment = ShaderProcessor::Compile(shaderData, "Fragment", "ps_5_0", fragmentPath);
			object = Shader::Create(vertex, fragment, options);
			ObjectDB::AllocateIdToGuid(object, guid, 1);
			AssetDB::SaveAssetObjectsToCache(std::vector<Object*> { object });
			BB_INFO("Shader \"" << GetName() << "\" imported and compiled from: " + path);
		}
		object->SetName(GetName());
		AddImportedObject(object, 1);
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
