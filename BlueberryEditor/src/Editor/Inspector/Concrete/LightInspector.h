#pragma once

#include "Editor\Inspector\ObjectInspector.h"

namespace Blueberry
{
	class LightInspector : public ObjectInspector
	{
	public:
		LightInspector() = default;
		virtual ~LightInspector() = default;

		virtual const char* GetIconPath(Object* object) override;
		virtual void DrawScene(Object* object) override;

	private:
		void DrawCone(const float& radius, const float& height, const int& mask);
	};
}