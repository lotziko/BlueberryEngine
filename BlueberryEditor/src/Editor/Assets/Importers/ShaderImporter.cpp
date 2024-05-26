#include "bbpch.h"
#include "ShaderImporter.h"

#include "Blueberry\Graphics\Shader.h"
#include "Editor\Assets\AssetDB.h"
#include "Editor\Assets\Processors\HLSLShaderParser.h"
#include "Editor\Assets\Processors\HLSLShaderProcessor.h"

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

		HLSLShaderProcessor vertexProcessor;
		HLSLShaderProcessor fragmentProcessor;

		Shader* object;
		if (AssetDB::HasAssetWithGuidInData(guid))
		{
			vertexProcessor.LoadBlob(vertexPath);
			fragmentProcessor.LoadBlob(fragmentPath);
			auto objects = AssetDB::LoadAssetObjects(guid, GetImportedObjects());
			if (objects.size() == 1 && objects[0].first->IsClassType(Shader::Type))
			{
				object = static_cast<Shader*>(objects[0].first);
				ObjectDB::AllocateIdToGuid(object, guid, objects[0].second);
				object->Initialize(vertexProcessor.GetShader(), fragmentProcessor.GetShader());
				object->SetState(ObjectState::Default);
				BB_INFO("Shader \"" << GetName() << "\" imported from cache.");
			}
		}
		else
		{
			std::string path = GetFilePath();
			std::string shaderCode;
			ShaderData data;

			if (HLSLShaderParser::Parse(path, shaderCode, data))
			{
				vertexProcessor.Compile(shaderCode, ShaderType::Vertex);
				fragmentProcessor.Compile(shaderCode, ShaderType::Fragment);
				vertexProcessor.SaveBlob(vertexPath);
				fragmentProcessor.SaveBlob(fragmentPath);

				object = Shader::Create(vertexProcessor.GetShader(), fragmentProcessor.GetShader(), data);
				ObjectDB::AllocateIdToGuid(object, guid, 1);
				AssetDB::SaveAssetObjectsToCache(std::vector<Object*> { object });
				BB_INFO("Shader \"" << GetName() << "\" imported and compiled from: " + path);
			}
		}
		object->SetName(GetName());
		AddImportedObject(object, 1);
		SetMainObject(1);
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
