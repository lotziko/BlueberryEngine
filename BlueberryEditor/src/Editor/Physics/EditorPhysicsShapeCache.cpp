#include "EditorPhysicsShapeCache.h"

#include "Blueberry\Core\ObjectDB.h"
#include "Blueberry\Graphics\Mesh.h"
#include "Blueberry\Tools\FileHelper.h"
#include "Blueberry\Tools\StringHelper.h"

#include "Editor\Path.h"

#include <fstream>

namespace Blueberry
{
	String GetPhysicsShapePath(const Guid& guid, FileId fileId, uint8_t key)
	{
		std::filesystem::path dataPath = Path::GetPhysicsShapeCachePath();
		dataPath.append(guid.ToString());
		if (!std::filesystem::exists(dataPath))
		{
			std::filesystem::create_directories(dataPath);
		}
		dataPath.append((std::to_string(fileId).append(std::to_string(key))).append(".shape"));
		return StringHelper::ToString(dataPath);
	}

	String GetPhysicsShapeDirectory(const Guid& guid)
	{
		std::filesystem::path dataPath = Path::GetPhysicsShapeCachePath();
		dataPath.append(guid.ToString());
		return StringHelper::ToString(dataPath);
	}
	
	uint8_t GetKey(bool isConvex, const Vector3& scale)
	{
		return (isConvex ? 1 : 0) | (scale.x > 0.0f ? 2 : 0) | (scale.y > 0.0f ? 4 : 0) | (scale.z > 0.0f ? 8 : 0);
	}

	void EditorPhysicsShapeCache::ClearImpl(Mesh* mesh)
	{
		auto& pair = ObjectDB::GetGuidAndFileIdFromObject(mesh);
		String directoryPath = GetPhysicsShapeDirectory(pair.first);
		if (std::filesystem::exists(directoryPath))
		{
			std::filesystem::remove_all(directoryPath);
		}
	}

	bool EditorPhysicsShapeCache::TryLoadImpl(Mesh* mesh, bool isConvex, const Vector3& scale, List<uint8_t>& data)
	{
		auto& pair = ObjectDB::GetGuidAndFileIdFromObject(mesh);
		String physicsShapePath = GetPhysicsShapePath(pair.first, pair.second, GetKey(isConvex, scale));
		if (std::filesystem::exists(physicsShapePath))
		{
			FileHelper::Load(data, physicsShapePath);
			return true;
		}
		return false;
	}

	void EditorPhysicsShapeCache::SaveImpl(Mesh* mesh, bool isConvex, const Vector3& scale, List<uint8_t>& data)
	{
		auto& pair = ObjectDB::GetGuidAndFileIdFromObject(mesh);
		String physicsShapePath = GetPhysicsShapePath(pair.first, pair.second, GetKey(isConvex, scale));
		FileHelper::Save(data, physicsShapePath);
	}
}
