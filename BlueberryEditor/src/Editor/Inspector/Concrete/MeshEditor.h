#pragma once

#include "Editor\Inspector\ObjectEditor.h"

namespace Blueberry
{
	class GfxTexture;

	class MeshEditor : public ObjectEditor
	{
	public:
		virtual ~MeshEditor() = default;

		virtual void OnEnable() override;
		virtual void OnDrawInspector() override;

	private:
		static inline GfxTexture* s_RenderTexture = nullptr;
	};
}