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
#include "Editor\Events\SceneEvents.h"

namespace Blueberry
{
	Scene* EditorSceneManager::s_EditorScene = nullptr;
	Scene* EditorSceneManager::s_RuntimeScene = nullptr;
	List<EditorSceneManager::PrefabSceneData> EditorSceneManager::s_PrefabScenes = {};
	String EditorSceneManager::s_Path = "";

	ObjectPtr<SceneSettings> EditorSceneManager::s_SceneSettings = nullptr;

	void EditorSceneManager::CreateEmpty(const String& path)
	{
		s_EditorScene = new Scene();
		Save();
		UpdateScene();
		SceneEvents::GetSceneLoaded().Invoke();
	}

	bool EditorSceneManager::HasScene()
	{
		return s_EditorScene != nullptr;
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
		if (s_RuntimeScene != nullptr)
		{
			return s_RuntimeScene;
		}
		return s_EditorScene;
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

		s_EditorScene = new Scene();

		s_Path = path;
		PlatformHelper::ShowProgressBar("Loading Scene", GetRelativePath());
		Deserialize(path);
		UpdateScene();
		PlatformHelper::HideProgressBar();
		SceneEvents::GetSceneLoaded().Invoke();
	}

	void EditorSceneManager::Reload()
	{
		if (s_EditorScene == nullptr && s_Path.empty())
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
		else if (s_EditorScene != nullptr)
		{
			Serialize(s_Path);
		}
	}

	void EditorSceneManager::Unload()
	{
		if (s_EditorScene != nullptr)
		{
			for (auto& pair : s_EditorScene->GetEntities())
			{
				Entity* entity = pair.second.Get();
				if (PrefabManager::IsPrefabInstanceRoot(entity))
				{
					Object::Destroy(PrefabManager::GetInstance(entity));
				}
			}
			s_EditorScene->Destroy();
			Entity::Poll();
			delete s_EditorScene;
			s_EditorScene = nullptr;
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
				SceneEvents::GetSceneLoaded().Invoke();
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
		SceneEvents::GetSceneLoaded().Invoke();
	}

	void EditorSceneManager::ClosePrefab(const bool& all)
	{
		for (size_t i = (all ? 0 : s_PrefabScenes.size() - 1); i < s_PrefabScenes.size(); ++i)
		{
			delete s_PrefabScenes[i].scene;
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
		SceneEvents::GetSceneLoaded().Invoke();
	}

	void EditorSceneManager::Run()
	{
		if (Application::IsRunning())
		{
			return;
		}
		Application::SetRunning(true);
		s_RuntimeScene = s_EditorScene->Clone();
		Entity::Poll();
	}

	void EditorSceneManager::Stop()
	{
		if (s_RuntimeScene == nullptr || !Application::IsRunning())
		{
			return;
		}
		s_RuntimeScene->Destroy();
		Entity::Poll();
		delete s_RuntimeScene;
		s_RuntimeScene = nullptr;
		Application::SetRunning(false);
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
			serializer.GatherPrefabs(s_EditorScene);
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
				s_EditorScene->AddEntity(entity);
			}
			else if (object->IsClassType(PrefabInstance::Type))
			{
				PrefabInstance* prefabInstance = static_cast<PrefabInstance*>(object);
				Entity* entity = prefabInstance->GetEntity();
				if (entity != nullptr)
				{
					s_EditorScene->AddEntity(entity);
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
			List<ObjectPtr<PrefabInstance>> prefabInstances;
			for (auto& pair : scene->GetEntities())
			{
				Entity* entity = pair.second.Get();
				if (PrefabManager::IsPrefabInstanceRoot(entity))
				{
					prefabInstances.push_back(PrefabManager::GetInstance(entity));
				}
			}
			for (const auto& prefabInstance : prefabInstances)
			{
				if (prefabInstance.IsValid())
				{
					prefabInstance->UpdateIfNeeded();
				}
			}
		}

		bool hasLighting = false;
		if (s_SceneSettings.IsValid())
		{
			LightingData* lightingData = s_SceneSettings->GetLightingData();
			if (lightingData != nullptr)
			{
				if (s_PrefabScenes.size() == 0 && s_EditorScene != nullptr)
				{
					hasLighting = true;
					lightingData->Apply();
				}
			}
			if (s_EditorScene == nullptr)
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
