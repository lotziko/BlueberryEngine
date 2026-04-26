#pragma once

#include "Editor\Inspector\ObjectEditor.h"

namespace Blueberry
{
	class SkinnedMeshRendererEditor : public ObjectEditor
	{
	public:
		virtual ~SkinnedMeshRendererEditor() = default;

		virtual void OnEnable() override;
		virtual void OnDrawInspector() override;

		virtual void OnDrawSceneSelected() override;

	private:
		SerializedProperty m_MeshProperty;
		SerializedProperty m_RootProperty;
		SerializedProperty m_MaterialsProperty;
		SerializedProperty m_IsCastingShadowsProperty;
	};
}