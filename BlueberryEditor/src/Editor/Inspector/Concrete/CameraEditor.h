#pragma once

#include "Editor\Inspector\ObjectEditor.h"

namespace Blueberry
{
	class CameraEditor : public ObjectEditor
	{
	public:
		CameraEditor();
		virtual ~CameraEditor() = default;

		virtual void OnEnable() override;
		virtual void OnDrawInspector() override;

		virtual Texture* GetIcon(Object* object) final;
		virtual void OnDrawSceneSelected() override;

	private:
		SerializedProperty m_IsOrthographicProperty;
		SerializedProperty m_OrthographicSizeProperty;
		SerializedProperty m_PixelSizeProperty;
		SerializedProperty m_FieldOfViewProperty;
		SerializedProperty m_AspectRatioProperty;
		SerializedProperty m_ZNearPlaneProperty;
		SerializedProperty m_ZFarPlaneProperty;
	};
}