#pragma once

#include "Blueberry\Events\Event.h"

namespace Blueberry
{
	class Scene;

	using SceneLoadEvent = Event<>;

	class EditorSceneManager
	{
	public:
		static Scene* GetScene();
		static void CreateEmpty(const String& path);
		static void Load(const String& path);
		static void Reload();
		static void Save();
		static void Unload();

		static void Run();
		static void Stop();
		static const bool& IsRunning();

		static SceneLoadEvent& GetSceneLoaded();

	private:
		static Scene* s_Scene;
		static String s_Path;
		static SceneLoadEvent s_SceneLoaded;
		static bool s_IsRunning;
	};
}