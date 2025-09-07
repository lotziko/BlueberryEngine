#pragma once

#include "Blueberry\Events\Event.h"
#include "Blueberry\Core\ObjectPtr.h"

namespace Blueberry
{
	class Scene;
	class SceneSettings;

	using SceneLoadEvent = Event<>;

	class EditorSceneManager
	{
	public:
		static Scene* GetScene();
		static const String& GetPath();
		static const Guid& GetGuid();
		static void CreateEmpty(const String& path);
		static void Load(const String& path);
		static void Reload();
		static void Save();
		static void Unload();

		static void Run();
		static void Stop();
		static const bool& IsRunning();

		static SceneLoadEvent& GetSceneLoaded();

		static SceneSettings* GetSettings();
		static void SetSettings(SceneSettings* settings);

	private:
		static void Serialize(const String& path);
		static void Deserialize(const String& path);

	private:
		static Scene* s_Scene;
		static String s_Path;
		static SceneLoadEvent s_SceneLoaded;
		static bool s_IsRunning;

		static inline ObjectPtr<SceneSettings> s_SceneSettings = {};
	};
}