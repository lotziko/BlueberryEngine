#pragma once

namespace Blueberry
{
	class Scene;

	class EditorSceneManager
	{
	public:
		static Ref<Scene> GetScene();
		static void CreateEmpty(const std::string& path);
		static void Load(const std::string& path);
		static void Save();

	private:
		static Ref<Scene> s_Scene;
	};
}