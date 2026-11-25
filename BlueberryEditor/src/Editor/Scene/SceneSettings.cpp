#include "SceneSettings.h"

#include "Blueberry\Core\ClassDB.h"
#include "Editor\EditorSceneManager.h"
#include "Editor\Assets\AssetDB.h"
#include "LightingData.h"

#include <filesystem>

namespace Blueberry
{
	OBJECT_DEFINITION(SceneSettings, Object)
	{
		DEFINE_FIELD(SceneSettings, m_LightingData, BindingType::ObjectPtr, FieldOptions().SetObjectType(&LightingData::Type))
	}

	LightingData* SceneSettings::GetLightingData()
	{
		if (!m_LightingData.IsValid())
		{
			m_LightingData = Object::Create<LightingData>();
			String lightingDataPath = EditorSceneManager::GetPath();
			lightingDataPath.replace(lightingDataPath.find(".scene"), 6, "\\LightingData.asset");
			auto relativeLightingDataPath = std::filesystem::relative(lightingDataPath, Path::GetAssetsPath());
			AssetDB::CreateAsset(m_LightingData.Get(), StringHelper::ToString(relativeLightingDataPath));
			AssetDB::SaveAssets();
		}

		return m_LightingData.Get();
	}
}