#include "bbpch.h"
#include "EditorSceneManager.h"

#include "Blueberry\Scene\Scene.h"

#include "Editor\Path.h"
#include "Editor\Serialization\YamlSerializer.h"
#include "Editor\Serialization\YamlHelper.h"

namespace Blueberry
{
	Scene* EditorSceneManager::s_Scene = nullptr;

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

	void EditorSceneManager::Load(const std::string& path)
	{
		if (s_Scene != nullptr)
		{
			s_Scene->Destroy();
		}

		s_Scene = new Scene();
		s_Scene->Initialize();

		YamlSerializer serializer;
		s_Scene->Deserialize(serializer, path);
	}

	void EditorSceneManager::Save()
	{
		std::filesystem::path scenePath = Path::GetAssetsPath();
		YamlSerializer serializer;
		s_Scene->Serialize(serializer, scenePath.append("Test.scene").string());
	}
}
