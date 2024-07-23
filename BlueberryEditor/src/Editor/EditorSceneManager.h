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
		static void CreateEmpty(const std::string& path);
		static void Load(const std::string& path);
		static void Save();

		static void Run();
		static void Stop();
		static const bool& IsRunning();

		static SceneLoadEvent GetSceneLoaded();

	private:
		static Scene* s_Scene;
		static std::string s_Path;
		static SceneLoadEvent s_SceneLoaded;
		static bool s_IsRunning;
	};
}