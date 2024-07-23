#pragma once

namespace Blueberry
{
	class Entity;

	class SceneInspector
	{
	public:
		SceneInspector();
		virtual ~SceneInspector();

		void DrawUI();
	};
}