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
		String physicsShapePath = ModelImporter::GetPhysicsShapePath(guid, fileId);
		if (std::filesystem::exists(physicsShapePath))
		{
			std::ifstream input;
			input.open(physicsShapePath.c_str(), std::ofstream::binary);
			if (input.is_open())
			{
				PhysicsShapeCache::Load(mesh, input);
				input.close();
			}
		}
		mesh->Apply();
	}
}
