#pragma once

#include "Editor\Inspector\ObjectEditor.h"

namespace ImGui
{
	class ClearOverrideEventArgs;
}

namespace Blueberry
{
	class TransformEditor : public ObjectEditor
	{
	public:
		virtual ~TransformEditor() = default;

		virtual void OnEnable() override;
		virtual void OnDisable() override;
		virtual void OnDrawInspector() override;
		virtual void OnDrawSceneSelected() override;

	private:
		void OnHierarchyUpdate();

		static void OnPropertyClear(ImGui::ClearOverrideEventArgs& args);

	private:
		SerializedProperty m_LocalPositionProperty;
		SerializedProperty m_LocalRotationProperty;
		SerializedProperty m_LocalScaleProperty;
		SerializedProperty m_LocalRotationEulerHintProperty;
		SerializedProperty m_IsStaticProperty;
	};
}