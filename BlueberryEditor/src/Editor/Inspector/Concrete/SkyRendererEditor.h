#pragma once

#include "Editor\Inspector\ObjectEditor.h"

namespace Blueberry
{
	class SkyRendererEditor : public ObjectEditor
	{
	public:
		SkyRendererEditor() = default;
		virtual ~SkyRendererEditor() = default;

		virtual void OnEnable() override;
		virtual void OnDrawInspector() override;

	private:
		SerializedProperty m_MaterialProperty;
		SerializedProperty m_ReflectionTextureProperty;
	};
}