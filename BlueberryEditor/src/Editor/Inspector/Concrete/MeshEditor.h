#pragma once

#include "Editor\Inspector\ObjectEditor.h"

namespace Blueberry
{
	class GfxTexture;

	class MeshEditor : public ObjectEditor
	{
	public:
		MeshEditor();
		virtual ~MeshEditor();

		virtual void OnDrawInspector() override;

	private:
		GfxTexture* m_RenderTexture;
	};
}