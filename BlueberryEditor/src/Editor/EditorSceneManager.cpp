#include "bbpch.h"
#include "EditorSceneManager.h"

#include "Blueberry\Scene\Scene.h"
#include "Blueberry\Serialization\YamlHelper.h"
#include "Blueberry\Serialization\Serializer.h"

#include "Editor\Path.h"

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

		std::filesystem::path scenePath = Path::GetAssetsPath();
		scenePath.append("Test.scene");

		ryml::Tree tree;
		YamlHelper::Load(tree, scenePath.string());
		ryml::NodeRef root = tree.rootref();
		Serializer serializer(root);

		s_Scene = CreateRef<Scene>();
		s_Scene->Initialize();

		s_Scene->Deserialize(serializer);
	}

	void EditorSceneManager::Save()
	{
		ryml::Tree tree;
		ryml::NodeRef root = tree.rootref();
		root |= ryml::MAP;
		Serializer serializer(root);
		s_Scene->Serialize(serializer);

		std::filesystem::path scenePath = Path::GetAssetsPath();
		scenePath.append("Test.scene");
		YamlHelper::Save(tree, scenePath.string());
	}
}
