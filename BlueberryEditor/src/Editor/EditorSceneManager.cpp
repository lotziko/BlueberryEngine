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
			else
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
			}
			else if (object.first->IsClassType(PrefabInstance::Type))
			{
				scene->AddEntity(((PrefabInstance*)object.first)->GetEntity());
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
				if (PrefabManager::IsPrefabInstace(entity))
				{
					Object::Destroy(entity);
				}
			}
			s_Scene->Destroy();
		}

		s_Scene = new Scene();
		s_Scene->Initialize();

		YamlSerializer serializer;
		Deserialize(s_Scene, serializer, path);
		s_Path = path;
	}

	void EditorSceneManager::Save()
	{
		YamlSerializer serializer;
		Serialize(s_Scene, serializer, s_Path);
	}
}
