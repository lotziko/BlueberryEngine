#pragma once

namespace Blueberry
{
	class Scene;

	class EditorSceneManager
	{
	public:
		static Scene* GetScene();
		static void CreateEmpty(const std::string& path);
		static void Load(const std::string& path);
		static void Save();

	private:
		static Scene* s_Scene;
	};
}