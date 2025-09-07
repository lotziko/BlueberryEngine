#pragma once

#include "Editor\Inspector\ObjectEditor.h"

namespace Blueberry
{
	class GfxTexture;

	class MaterialEditor : public ObjectEditor
	{
	public:
		MaterialEditor();
		virtual ~MaterialEditor();

	private:
		GfxTexture* m_RenderTexture;
	};
}