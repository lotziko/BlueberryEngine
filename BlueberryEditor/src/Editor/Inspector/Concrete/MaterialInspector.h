#pragma once

#include "Editor\Inspector\ObjectInspector.h"

namespace Blueberry
{
	class GfxTexture;

	class MaterialInspector : public ObjectInspector
	{
	public:
		MaterialInspector();
		virtual ~MaterialInspector();

		virtual void Draw(Object* object) override;

	private:
		GfxTexture* m_RenderTexture;
	};
}