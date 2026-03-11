#include "AssetFinalizer.h"

#include "Blueberry\Graphics\Texture2D.h"
#include "Blueberry\Graphics\TextureCube.h"
#include "Blueberry\Graphics\Texture3D.h"
#include "Blueberry\Graphics\Mesh.h"
#include "Blueberry\Graphics\Shader.h"
#include "Blueberry\Graphics\ComputeShader.h"
#include "Blueberry\Tools\FileHelper.h"

#include "Editor\Assets\Processors\HLSLShaderProcessor.h"
#include "Editor\Assets\Processors\HLSLComputeShaderProcessor.h"
#include "Editor\Assets\Importers\ShaderImporter.h"
#include "Editor\Assets\Importers\ComputeShaderImporter.h"
#include "Editor\Assets\Importers\TextureImporter.h"

namespace Blueberry
{
	void AssetFinalizer::Finalize(Object* object, const Guid& guid, const FileId& fileId)
	{
		TypeId type = object->GetType();
		if (type == Mesh::Type)
		{
			Mesh* mesh = static_cast<Mesh*>(object);
			mesh->Apply();
		}
		else if (type == Texture2D::Type)
		{
			Texture2D* texture = static_cast<Texture2D*>(object);
			String texturePath = TextureImporter::GetTexturePath(guid);
			if (std::filesystem::exists(texturePath))
			{
				uint8_t* data;
				size_t length;
				FileHelper::Load(data, length, texturePath);
				texture->SetData(data, length);
			}
			texture->Apply();
		}
		else if (type == TextureCube::Type)
		{
			TextureCube* texture = static_cast<TextureCube*>(object);
			String texturePath = TextureImporter::GetTexturePath(guid);
			if (std::filesystem::exists(texturePath))
			{
				uint8_t* data;
				size_t length;
				FileHelper::Load(data, length, texturePath);
				texture->SetData(data, length);
			}
			texture->Apply();
		}
		else if (type == Texture3D::Type)
		{
			Texture3D* texture = static_cast<Texture3D*>(object);
			String texturePath = TextureImporter::GetTexturePath(guid);
			if (std::filesystem::exists(texturePath))
			{
				uint8_t* data;
				size_t length;
				FileHelper::Load(data, length, texturePath);
				texture->SetData(data, length);
			}
			texture->Apply();
		}
		else if (type == Shader::Type)
		{
			Shader* shader = static_cast<Shader*>(object);
			String folderPath = ShaderImporter::GetShaderFolder(guid);
			HLSLShaderProcessor processor;
			if (processor.LoadVariants(folderPath))
			{
				shader->Initialize(processor.GetVariantsData());
			}
		}
		else if (type == ComputeShader::Type)
		{
			ComputeShader* shader = static_cast<ComputeShader*>(object);
			String folderPath = ComputeShaderImporter::GetShaderFolder(guid);
			HLSLComputeShaderProcessor processor;
			if (processor.LoadKernels(folderPath))
			{
				shader->Initialize(processor.GetShaders());
			}
		}
		object->SetState(ObjectState::Default);
	}
}
