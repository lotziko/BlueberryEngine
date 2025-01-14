#include "bbpch.h"
#include "EditorSceneManager.h"

#include "Blueberry\Scene\Scene.h"

#include "Editor\Path.h"
#include "Editor\Assets\AssetDB.h"
#include "Editor\Assets\AssetImporter.h"
#include "Editor\Serialization\YamlSerializer.h"
#include "Editor\Serialization\YamlHelper.h"
#include "Editor\Prefabs\PrefabManager.h"
#include "Editor\Prefabs\PrefabInstance.h"

namespace Blueberry
{
	Scene* EditorSceneManager::s_Scene = nullptr;
	std::string EditorSceneManager::s_Path = "";
	SceneLoadEvent EditorSceneManager::s_SceneLoaded = {};
	bool EditorSceneManager::s_IsRunning = false;

	void EditorSceneManager::CreateEmpty(const std::string& path)
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

	void Serialize(Scene* scene, Serializer& serializer, const std::string& path)
	{
		for (auto& pair : scene->GetEntities())
		{
			// Components are being added automatically
			Entity* entity = pair.second.Get();
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

	void Deserialize(Scene* scene, Serializer& serializer, const std::string& path)
	{
		serializer.Deserialize(path);
		for (auto& object : serializer.GetDeserializedObjects())
		{
			if (object.first->IsClassType(Entity::Type))
			{
				Entity* entity = (Entity*)object.first;
				scene->AddEntity(entity);
				entity->OnCreate();

			}
			else if (object.first->IsClassType(PrefabInstance::Type))
			{
				PrefabInstance* prefabInstance = (PrefabInstance*)object.first;
				prefabInstance->OnCreate();
				Entity* entity = prefabInstance->GetEntity();
				scene->AddEntity(entity);
				entity->OnCreate();
			}
		}
	}

	void EditorSceneManager::Load(const std::string& path)
	{
		if (s_Scene != nullptr)
		{
			for (auto& pair : s_Scene->GetEntities())
			{
				Entity* entity = pair.second.Get();
				if (PrefabManager::IsPartOfPrefabInstance(entity))
				{
					Object::Destroy(PrefabManager::GetInstance(entity));
				}
			}
			s_Scene->Destroy();
		}

		s_Scene = new Scene();
		s_Scene->Initialize();

		YamlSerializer serializer;
		Deserialize(s_Scene, serializer, path);
		s_Path = path;
		s_SceneLoaded.Invoke();
	}

	void EditorSceneManager::Save()
	{
		YamlSerializer serializer;
		Serialize(s_Scene, serializer, s_Path);
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
}
