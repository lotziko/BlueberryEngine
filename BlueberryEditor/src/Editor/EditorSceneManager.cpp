#include "EditorSceneManager.h"

#include "Blueberry\Core\Application.h"
#include "Blueberry\Core\ObjectCloner.h"
#include "Blueberry\Scene\Scene.h"
#include "Blueberry\Scene\Components\Transform.h"

#include "Editor\Path.h"
#include "Editor\Assets\AssetDB.h"
#include "Editor\Assets\AssetImporter.h"
#include "Editor\Serialization\EditorSerializer.h"
#include "Editor\Prefabs\PrefabManager.h"
#include "Editor\Prefabs\PrefabInstance.h"
#include "Editor\Scene\SceneSettings.h"
#include "Editor\Scene\LightingData.h"
#include "Editor\Misc\PlatformHelper.h"

namespace Blueberry
{
	Scene* EditorSceneManager::s_Scene = nullptr;
	List<EditorSceneManager::PrefabSceneData> EditorSceneManager::s_PrefabScenes = {};
	String EditorSceneManager::s_Path = "";
	SceneLoadEvent EditorSceneManager::s_SceneLoaded = {};

	ObjectPtr<SceneSettings> EditorSceneManager::s_SceneSettings = nullptr;

	void EditorSceneManager::CreateEmpty(const String& path)
	{
		s_Scene = new Scene();
		s_Scene->Initialize();
		Save();
		UpdateScene();
		s_SceneLoaded.Invoke();
	}

	bool EditorSceneManager::HasScene()
	{
		return s_Scene != nullptr;
	}

	bool EditorSceneManager::HasPrefabScene()
	{
		return s_PrefabScenes.size() > 0;
	}

	Scene* EditorSceneManager::GetScene()
	{
		if (s_PrefabScenes.size() > 0)
		{
			return s_PrefabScenes[s_PrefabScenes.size() - 1].scene;
		}
		return s_Scene;
	}

	const String EditorSceneManager::GetRelativePath()
	{
		return StringHelper::ToString(std::filesystem::relative(s_Path, Path::GetAssetsPath()));
	}

	const String& EditorSceneManager::GetPath()
	{
		return s_Path;
	}

	const Guid& EditorSceneManager::GetGuid()
	{
		String path = StringHelper::ToString(std::filesystem::relative(s_Path, Path::GetAssetsPath()));
		return AssetDB::GetImporter(path)->GetGuid();
	}

	void EditorSceneManager::Load(const String& path)
	{
		if (s_PrefabScenes.size() > 0)
		{
			ClosePrefab(true);
		}

		Unload();
		Entity::Poll();

		s_Scene = new Scene();
		s_Scene->Initialize();

		s_Path = path;
		PlatformHelper::ShowProgressBar("Loading Scene", GetRelativePath());
		Deserialize(path);
		UpdateScene();
		PlatformHelper::HideProgressBar();
		s_SceneLoaded.Invoke();
	}

	void EditorSceneManager::Reload()
	{
		if (s_Scene == nullptr && s_Path.empty())
		{
			return;
		}
		Load(s_Path);
	}

	void EditorSceneManager::Save()
	{
		if (s_PrefabScenes.size() > 0)
		{
			EditorSceneManager::PrefabSceneData& data = s_PrefabScenes[s_PrefabScenes.size() - 1];
			String relativePath = AssetDB::GetRelativeAssetPath(data.root);
			std::filesystem::path dataPath = Path::GetAssetsPath();
			dataPath.append(relativePath);
			Serialize(StringHelper::ToString(dataPath));

			// TODO recursive dependencies
			HashSet<Guid> dependent;
			AssetDB::GetDependent(ObjectDB::GetGuidFromObject(data.root), dependent);
			for (Guid guid : dependent)
			{
				AssetDB::ImportAsset(AssetDB::GetRelativePath(guid));
			}
		}
		else if (s_Scene != nullptr)
		{
			Serialize(s_Path);
		}
	}

	void EditorSceneManager::Unload()
	{
		if (s_Scene != nullptr)
		{
			for (auto& pair : s_Scene->GetEntities())
			{
				Entity* entity = pair.second.Get();
				if (PrefabManager::IsPrefabInstanceRoot(entity))
				{
					Object::Destroy(PrefabManager::GetInstance(entity));
				}
			}
			s_Scene->Destroy();
			s_Scene = nullptr;
			UpdateScene();
		}
	}

	void EditorSceneManager::OpenPrefab(Entity*& root)
	{
		// TODO open nested prefabs, for now close all
		ClosePrefab(true);
		for (size_t i = 0; i < s_PrefabScenes.size(); ++i)
		{
			if (s_PrefabScenes[i].root == root)
			{
				s_PrefabScenes.move_element(i, s_PrefabScenes.size() - 1);
				s_SceneLoaded.Invoke();
				return;
			}
		}

		EditorSceneManager::PrefabSceneData data = {};
		data.root = root;
		data.scene = new Scene();

		String relativePath = AssetDB::GetRelativeAssetPath(root);
		std::filesystem::path dataPath = Path::GetAssetsPath();
		dataPath.append(relativePath);

		EditorSerializer serializer = {};
		serializer.Deserialize(StringHelper::ToString(dataPath), SerializationFlags::EditorOnly | SerializationFlags::Text | SerializationFlags::HasHeaders);
		serializer.FinalizeObjects();
		for (auto& pair : serializer.GetDeserializedObjects())
		{
			Object* object = ObjectDB::GetObject(pair.first);
			if (object->IsClassType(Entity::Type) && !PrefabManager::IsPartOfPrefabInstance(object))
			{
				Entity* entity = static_cast<Entity*>(object);
				if (entity->GetTransform()->GetParent() == nullptr)
				{
					data.scene->AddEntity(entity);
				}
			}
			else if (object->IsClassType(PrefabInstance::Type))
			{
				PrefabInstance* prefabInstance = static_cast<PrefabInstance*>(object);
				Entity* entity = prefabInstance->GetEntity();
				if (entity != nullptr)
				{
					data.scene->AddEntity(entity);
				}
			}
		}

		s_PrefabScenes.push_back(std::move(data));
		UpdateScene();
		s_SceneLoaded.Invoke();
	}

	void EditorSceneManager::ClosePrefab(const bool& all)
	{
		for (size_t i = (all ? 0 : s_PrefabScenes.size() - 1); i < s_PrefabScenes.size(); ++i)
		{
			s_PrefabScenes[i].scene->Destroy();
		}
		if (all)
		{
			s_PrefabScenes.clear();
		}
		else
		{
			s_PrefabScenes.pop_back();
		}
		UpdateScene();
		s_SceneLoaded.Invoke();
	}

	void EditorSceneManager::Run()
	{
		if (Application::IsRunning())
		{
			return;
		}
		Unload();
		Entity::Poll();
		Application::SetRunning(true);
		Reload();
	}

	void EditorSceneManager::Stop()
	{
		if (s_Scene == nullptr || !Application::IsRunning())
		{
			return;
		}
		Unload();
		Entity::Poll();
		Application::SetRunning(false);
		Reload();
	}

	SceneLoadEvent& EditorSceneManager::GetSceneLoaded()
	{
		return s_SceneLoaded;
	}

	SceneSettings* EditorSceneManager::GetSettings()
	{
		if (!s_SceneSettings.IsValid())
		{
			s_SceneSettings = Object::Create<SceneSettings>();
		}
		return s_SceneSettings.Get();
	}

	void EditorSceneManager::SetSettings(SceneSettings* settings)
	{
		s_SceneSettings = settings;
	}

	void EditorSceneManager::Serialize(const String& path)
	{
		bool isPrefab = s_PrefabScenes.size() > 0;
		EditorSerializer serializer = {};
		if (isPrefab)
		{
			auto& sceneData = s_PrefabScenes[s_PrefabScenes.size() - 1];
			serializer.GatherPrefabs(sceneData.scene);
			// This breaks prefabs by cloning PrefabInstance and some components on save
			//serializer.SetGuid(ObjectDB::GetGuidFromObject(sceneData.root));
		}
		else
		{
			if (s_SceneSettings.IsValid())
			{
				serializer.AddObject(s_SceneSettings.Get());
			}
			serializer.GatherPrefabs(s_Scene);
		}
		serializer.Serialize(path, SerializationFlags::EditorOnly | SerializationFlags::Text | SerializationFlags::HasHeaders);
		AssetDB::Refresh();	// maybe skip refresh on prefabs and save straight to cache?
	}

	void EditorSceneManager::Deserialize(const String& path)
	{
		EditorSerializer serializer = {};
		serializer.Deserialize(path, SerializationFlags::EditorOnly | SerializationFlags::Text | SerializationFlags::HasHeaders);
		serializer.FinalizeObjects();
		for (auto& pair : serializer.GetDeserializedObjects())
		{
			Object* object = ObjectDB::GetObject(pair.first);
			if (object->IsClassType(Entity::Type))
			{
				Entity* entity = static_cast<Entity*>(object);
				s_Scene->AddEntity(entity);
			}
			else if (object->IsClassType(PrefabInstance::Type))
			{
				PrefabInstance* prefabInstance = static_cast<PrefabInstance*>(object);
				Entity* entity = prefabInstance->GetEntity();
				if (entity != nullptr)
				{
					s_Scene->AddEntity(entity);
				}
			}
			else if (object->IsClassType(SceneSettings::Type))
			{
				s_SceneSettings = static_cast<SceneSettings*>(object);
			}
		}
		Entity::Poll();
	}

	void EditorSceneManager::UpdateScene()
	{
		Scene* scene = GetScene();
		if (scene != nullptr)
		{
			for (auto& pair : scene->GetEntities())
			{
				Entity* entity = pair.second.Get();
				if (PrefabManager::IsPrefabInstanceRoot(entity))
				{
					PrefabInstance* instance = PrefabManager::GetInstance(entity);
					instance->UpdateIfNeeded();
				}
			}
		}

		bool hasLighting = false;
		if (s_SceneSettings.IsValid())
		{
			LightingData* lightingData = s_SceneSettings->GetLightingData();
			if (lightingData != nullptr)
			{
				if (s_PrefabScenes.size() == 0 && s_Scene != nullptr)
				{
					hasLighting = true;
					lightingData->Apply();
				}
			}
			if (s_Scene == nullptr)
			{
				Object::Destroy(s_SceneSettings.Get());
				s_SceneSettings = nullptr;
			}
		}
		if (!hasLighting)
		{
			LightingData::Clear();
		}
	}
}
