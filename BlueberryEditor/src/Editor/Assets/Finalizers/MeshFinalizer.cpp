#include "MeshFinalizer.h"

#include "Blueberry\Graphics\Mesh.h"
#include "Blueberry\Physics\PhysicsShapeCache.h"
#include "Editor\Assets\Importers\ModelImporter.h"

#include <fstream>
#include <filesystem>

namespace Blueberry
{
	void MeshFinalizer::Finalize(Object* object, const Guid& guid, const FileId& fileId)
	{
		Mesh* mesh = static_cast<Mesh*>(object);
		mesh->Apply();
	}
}
