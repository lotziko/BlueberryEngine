#pragma once

#include "Blueberry\Events\Event.h"
#include "Blueberry\Core\ObjectPtr.h"

namespace Blueberry
{
	class Scene;
	class SceneSettings;
	class Entity;

	using SceneLoadEvent = Event<>;

	class EditorSceneManager
	{
	public:
		static bool HasScene();
		static bool HasPrefabScene();
		static Scene* GetScene();
		static const String GetRelativePath();
		static const String& GetPath();
		static const Guid& GetGuid();
		static void CreateEmpty(const String& path);
		static void Load(const String& path);
		static void Reload();
		static void Save();
		static void Unload();

		static void OpenPrefab(Entity*& root);
		static void ClosePrefab(const bool& all);

		static void Run();
		static void Stop();

		static SceneLoadEvent& GetSceneLoaded();

		static SceneSettings* GetSettings();
		static void SetSettings(SceneSettings* settings);

	private:
		static void Serialize(const String& path);
		static void Deserialize(const String& path);
		static void UpdateScene();

	private:
		struct PrefabSceneData
		{
			Scene* scene;
			Entity* root;
		};

	private:
		static Scene* s_Scene;
		static List<PrefabSceneData> s_PrefabScenes;
		static String s_Path;
		static SceneLoadEvent s_SceneLoaded;

		static inline ObjectPtr<SceneSettings> s_SceneSettings = {};
	};
}