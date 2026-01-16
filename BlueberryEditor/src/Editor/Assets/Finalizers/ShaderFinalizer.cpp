#include "ShaderFinalizer.h"

#include "Blueberry\Graphics\Shader.h"
#include "Editor\Assets\Processors\HLSLShaderProcessor.h"
#include "Editor\Assets\Importers\ShaderImporter.h"

namespace Blueberry
{
	void ShaderFinalizer::Finalize(Object* object, const Guid& guid, const FileId& fileId)
	{
		Shader* shader = static_cast<Shader*>(object);
		String folderPath = ShaderImporter::GetShaderFolder(guid);
		HLSLShaderProcessor processor;
		if (processor.LoadVariants(folderPath))
		{
			shader->Initialize(processor.GetVariantsData());
		}
	}
}
