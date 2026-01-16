#include "ComputeShaderFinalizer.h"

#include "Blueberry\Graphics\Shader.h"
#include "Editor\Assets\Processors\HLSLComputeShaderProcessor.h"
#include "Editor\Assets\Importers\ComputeShaderImporter.h"

namespace Blueberry
{
	void ComputeShaderFinalizer::Finalize(Object* object, const Guid& guid, const FileId& fileId)
	{
		ComputeShader* shader = static_cast<ComputeShader*>(object);
		String folderPath = ComputeShaderImporter::GetShaderFolder(guid);
		HLSLComputeShaderProcessor processor;
		if (processor.LoadKernels(folderPath))
		{
			shader->Initialize(processor.GetShaders());
		}
	}
}
