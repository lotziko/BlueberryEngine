#pragma once

#include "Editor\Panels\EditorWindow.h"
#include "Editor\Serialization\SerializedObject.h"

namespace Blueberry
{
	class Entity;
	class ObjectEditor;

	class SceneInspector : public EditorWindow
	{
		OBJECT_DECLARATION(SceneInspector)

	public:
		SceneInspector();
		virtual ~SceneInspector();

		static void Open();

		virtual void OnDrawUI() final;

	private:
		void SelectionChanged();

	private:
		ObjectEditor* m_Editor;
		bool m_IsInvalidSelection;
	};
}