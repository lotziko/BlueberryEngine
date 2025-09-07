#include "EditorSceneManager.h"

#include "Blueberry\Scene\Scene.h"
#include "Blueberry\Scene\Components\SpriteRenderer.h"

#include "Editor\Path.h"
#include "Editor\Assets\AssetDB.h"
#include "Editor\Assets\AssetImporter.h"
#include "Editor\Serialization\YamlSerializer.h"
#include "Editor\Serialization\YamlHelper.h"
#include "Editor\Prefabs\PrefabManager.h"
#include "Editor\Prefabs\PrefabInstance.h"
#include "Editor\Scene\SceneSettings.h"
#include "Editor\Scene\LightingData.h"

namespace Blueberry
{
	Scene* EditorSceneManager::s_Scene = nullptr;
	String EditorSceneManager::s_Path = "";
	SceneLoadEvent EditorSceneManager::s_SceneLoaded = {};
	bool EditorSceneManager::s_IsRunning = false;

	void EditorSceneManager::CreateEmpty(const String& path)
	{
		s_Scene = new Scene();
		s_Scene->Initialize();

		for (int i = 0; i < 2; i++)
		{
			auto test = s_Scene->CreateEntity("Test");
			test->AddComponent<SpriteRenderer>();
		}

		Save();
		s_SceneLoaded.Invoke();
	}

	Scene* EditorSceneManager::GetScene()
	{
		return s_Scene;
	}

	const String& EditorSceneManager::GetPath()
	{
		return s_Path;
	}

	const Guid& EditorSceneManager::GetGuid()
	{
		std::string path = std::filesystem::relative(s_Path, Path::GetAssetsPath()).string();
		return AssetDB::GetImporter(String(path.data()))->GetGuid();
	}

	void EditorSceneManager::Load(const String& path)
	{
		Unload();

		s_Scene = new Scene();
		s_Scene->Initialize();

		s_Path = path;
		Deserialize(path);

		s_SceneLoaded.Invoke();
	}

	void EditorSceneManager::Reload()
	{
		if (s_Scene == nullptr)
		{
			return;
		}
		Load(s_Path);
	}

	void EditorSceneManager::Save()
	{
		if (s_Scene != nullptr)
		{
			Serialize(s_Path);
		}
	}

	void EditorSceneManager::Unload()
	{
		if (s_Scene != nullptr)
		{
			for (auto& rootEntity : s_Scene->GetRootEntities())
			{
				Entity* entity = rootEntity.Get();
				if (PrefabManager::IsPartOfPrefabInstance(entity))
				{
					Object::Destroy(PrefabManager::GetInstance(entity));
				}
			}
			s_Scene->Destroy();
			if (s_SceneSettings.IsValid())
			{
				Object::Destroy(s_SceneSettings.Get());
				s_SceneSettings = nullptr;
			}
		}
	}

	void EditorSceneManager::Run()
	{
		if (s_IsRunning)
		{
			return;
		}
		s_IsRunning = true;
	}

	void EditorSceneManager::Stop()
	{
		if (s_Scene == nullptr || !s_IsRunning)
		{
			return;
		}
		Load(s_Path);
		s_IsRunning = false;
	}

	const bool& EditorSceneManager::IsRunning()
	{
		return s_IsRunning;
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
		YamlSerializer serializer;
		if (s_SceneSettings.IsValid())
		{
			serializer.AddObject(s_SceneSettings.Get());
		}
		for (auto& rootEntity : s_Scene->GetRootEntities())
		{
			// Components are being added automatically
			Entity* entity = rootEntity.Get();
			PrefabInstance* prefabInstance = PrefabManager::GetInstance(entity);
			if (prefabInstance != nullptr)
			{
				serializer.AddObject(prefabInstance);
			}
			else if (!PrefabManager::IsPartOfPrefabInstance(entity))
			{
				serializer.AddObject(entity);
			}
		}
		serializer.Serialize(path);
	}

	void EditorSceneManager::Deserialize(const String& path)
	{
		YamlSerializer serializer;
		serializer.Deserialize(path);
		for (auto& object : serializer.GetDeserializedObjects())
		{
			if (object.first->IsClassType(Entity::Type))
			{
				Entity* entity = static_cast<Entity*>(object.first);
				s_Scene->AddEntity(entity);
				entity->OnCreate();
			}
			else if (object.first->IsClassType(PrefabInstance::Type))
			{
				PrefabInstance* prefabInstance = static_cast<PrefabInstance*>(object.first);
				prefabInstance->OnCreate();
				Entity* entity = prefabInstance->GetEntity();
				s_Scene->AddEntity(entity);
				entity->OnCreate();
			}
			else if (object.first->IsClassType(SceneSettings::Type))
			{
				s_SceneSettings = static_cast<SceneSettings*>(object.first);
			}
		}
		if (s_SceneSettings.IsValid())
		{
			LightingData* lightingData = s_SceneSettings->GetLightingData();
			if (lightingData != nullptr)
			{
				lightingData->Apply(s_Scene);
			}
		}
	}
}
