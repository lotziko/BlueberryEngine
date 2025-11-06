#pragma once

#include "Editor\Inspector\ObjectEditor.h"

namespace Blueberry
{
	class GfxTexture;

	class MaterialEditor : public ObjectEditor
	{
	public:
		virtual ~MaterialEditor() = default;

		virtual void OnEnable() override;
		virtual void OnDrawInspector() override;

	private:
		static inline GfxTexture* s_RenderTexture = nullptr;

		SerializedProperty m_ShaderProperty;
		SerializedProperty m_TexturesProperty;
	};
}