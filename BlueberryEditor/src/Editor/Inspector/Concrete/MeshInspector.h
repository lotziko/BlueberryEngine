#pragma once

#include "Editor\Inspector\ObjectInspector.h"

namespace Blueberry
{
	class GfxTexture;

	class MeshInspector : public ObjectInspector
	{
	public:
		MeshInspector();
		virtual ~MeshInspector();

		virtual void Draw(Object* object) override;

	private:
		GfxTexture* m_RenderTexture;
	};
}