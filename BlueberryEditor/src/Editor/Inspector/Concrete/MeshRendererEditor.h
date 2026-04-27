#pragma once

#include "Editor\Inspector\ObjectEditor.h"

namespace Blueberry
{
	class MeshRendererEditor : public ObjectEditor
	{
	public:
		virtual ~MeshRendererEditor() = default;

		virtual void OnEnable() override;
		virtual void OnDrawInspector() override;

		virtual void OnDrawSceneSelected() override;

	private:
		SerializedProperty m_MeshProperty;
		SerializedProperty m_MaterialsProperty;
		SerializedProperty m_IsCastingShadowsProperty;
		SerializedProperty m_IsBakeableProperty;
	};
}