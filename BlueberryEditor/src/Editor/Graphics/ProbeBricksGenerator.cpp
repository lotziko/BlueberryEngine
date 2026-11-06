#include "ProbeBricksGenerator.h"

#include "Blueberry\Graphics\Texture3D.h"
#include "Blueberry\Scene\Components\ProbeVolume.h"
#include "Blueberry\Scene\Components\Transform.h"
#include "Editor\EditorSceneManager.h"
#include "Editor\Assets\AssetDB.h"
#include "Editor\Scene\SceneSettings.h"
#include "Editor\Scene\LightingData.h"
#include "Baking\LightmappingManager.h"

#include <filesystem>

namespace Blueberry
{
	void ProbeBricksGenerator::GenerateProbeVolumeBricks(ProbeVolume* probeVolume)
	{
		EditorSceneManager::Save();
		Vector3 position = probeVolume->GetTransform()->GetPosition();
		Vector3 size = probeVolume->GetSize();
		Vector3 minPosition = position - size / 2;
		Vector3Int probeVolumeSize = probeVolume->GetDensity();
		Vector3 brickSize = Vector3(size.x / probeVolumeSize.x, size.y / probeVolumeSize.y, size.z / probeVolumeSize.z);

		List<ProbeVolumeBrickData> bricks;
		Dictionary<Vector3Int, uint64_t> positionToIndex;
		List<Vector3> positions;

		for (int z = 0; z < probeVolumeSize.z; ++z)
		{
			for (int y = 0; y < probeVolumeSize.y; ++y)
			{
				for (int x = 0; x < probeVolumeSize.x; ++x)
				{
					ProbeVolumeBrickData brick = {};
					Vector3 brickMinPosition = minPosition + brickSize * Vector3(x, y, z);
					Vector3 brickMaxPosition = brickMinPosition + brickSize;
					for (uint32_t i = 0; i < 64; ++i)
					{
						uint32_t xi = i % 4;
						uint32_t yi = (i / 4) % 4;
						uint32_t zi = i / 16;

						Vector3Int intProbePosition = Vector3Int(3 * x + xi, 3 * y + yi, 3 * z + zi);	// TODO change for max sparse level
						auto it = positionToIndex.find(intProbePosition);
						if (it != positionToIndex.end())
						{
							brick.SetColorOffset(i, it->second);
						}
						else
						{
							size_t index = positions.size();
							Vector3 probePosition = Vector3(brickMinPosition.x + (brickMaxPosition.x - brickMinPosition.x) * (xi / 3.0f), brickMinPosition.y + (brickMaxPosition.y - brickMinPosition.y) * (yi / 3.0f), brickMinPosition.z + (brickMaxPosition.z - brickMinPosition.z) * (zi / 3.0f));
							brick.SetColorOffset(i, index);
							positionToIndex.insert_or_assign(intProbePosition, index);
							positions.push_back(probePosition);
						}
					}
					bricks.push_back(std::move(brick));
				}
			}
		}

		CalculationParams params = {};
		params.probeVolumeSize = probeVolumeSize;
		params.tileSize = 256;
		params.samplePerTexel = 10000;

		LightmappingManager::BakeProbes(EditorSceneManager::GetScene(), params, positions);
		while (LightmappingManager::GetLightmappingState() != LightmappingState::Waiting)
		{
		}
		CalculationResult& result = LightmappingManager::GetCalculationResult();
		LightmappingManager::Shutdown();

		LightingData* lightingData = EditorSceneManager::GetSettings()->GetLightingData();
		uint32_t index = probeVolume->GetIndex();
		if (index == UINT32_MAX)
		{
			index = std::max(static_cast<uint32_t>(lightingData->GetProbeVolumeCount()), 1u);
		}

		String path = EditorSceneManager::GetPath();
		String name = "\\ProbeVolume";
		name.append(std::to_string(index));
		name.append(".asset");
		path.replace(path.find(".scene"), 6, name);
		std::filesystem::create_directories(std::filesystem::path(path).parent_path());

		ProbeVolumeInfo* info = Object::Create<ProbeVolumeInfo>();
		info->SetSize(probeVolumeSize);
		info->SetBricks(bricks);
		info->SetColors(result.probeOutput);

		auto relativePath = std::filesystem::relative(path, Path::GetAssetsPath());
		AssetDB::CreateAsset(info, StringHelper::ToString(relativePath));
		AssetDB::Refresh();

		probeVolume->SetIndex(index);
		lightingData->SetProbeVolume(index, probeVolume, info);
		lightingData->Apply();
		AssetDB::SetDirty(lightingData);
		AssetDB::SaveAssets();
	}
}