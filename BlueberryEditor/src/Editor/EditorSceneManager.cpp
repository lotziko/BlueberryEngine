#include "bbpch.h"
#include "EditorSceneManager.h"

#include "Blueberry\Scene\Scene.h"

#include "Editor\Path.h"
#include "Editor\Serialization\YamlSerializer.h"
#include "Editor\Serialization\YamlHelper.h"

namespace Blueberry
{
	Ref<Scene> EditorSceneManager::s_Scene = nullptr;

	void EditorSceneManager::CreateEmpty(const std::string& path)
	{
		s_Scene = CreateRef<Scene>();
		s_Scene->Initialize();

		for (int i = 0; i < 2; i++)
		{
			auto test = s_Scene->CreateEntity("Test");
			test->AddComponent<SpriteRenderer>();
		}

		Save();
	}

	Ref<Scene> EditorSceneManager::GetScene()
	{
		return s_Scene;
	}

	void EditorSceneManager::Load(const std::string& path)
	{
		if (s_Scene != nullptr)
		{
			s_Scene->Destroy();
		}

		s_Scene = CreateRef<Scene>();
		s_Scene->Initialize();

		YamlSerializer serializer;
		std::filesystem::path scenePath = Path::GetAssetsPath();
		s_Scene->Deserialize(serializer, scenePath.append("Test.scene").string());
	}

	void EditorSceneManager::Save()
	{
		std::filesystem::path scenePath = Path::GetAssetsPath();
		YamlSerializer serializer;
		s_Scene->Serialize(serializer, scenePath.append("Test.scene").string());
	}
}
