#pragma once

#include "Editor\Inspector\ObjectInspector.h"

namespace Blueberry
{
	class RenderTexture;

	class MeshInspector : public ObjectInspector
	{
	public:
		MeshInspector();
		virtual ~MeshInspector();

		virtual void Draw(Object* object) override;

	private:
		RenderTexture* m_RenderTexture;
	};
}