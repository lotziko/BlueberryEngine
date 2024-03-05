#pragma once

namespace Blueberry
{
	class Entity;

	class SceneInspector
	{
	public:
		SceneInspector() = default;
		virtual ~SceneInspector() = default;

		void DrawUI();
	};
}