#pragma once

#include "Editor\Panels\EditorWindow.h"

namespace Blueberry
{
	class Entity;

	class SceneInspector : public EditorWindow
	{
		OBJECT_DECLARATION(SceneInspector)

	public:
		SceneInspector();
		virtual ~SceneInspector();

		static void Open();

		virtual void OnDrawUI() final;
	};
}