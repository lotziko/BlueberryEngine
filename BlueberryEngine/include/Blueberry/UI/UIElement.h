#pragma once

#include "Blueberry\Core\Base.h"

namespace Rml
{
	class Element;
}

namespace Blueberry
{
	class BB_API UIElement
	{
	public:
		UIElement GetElementById(const char* name);
		void SetProperty(const char* name, const char* value);

	private:
		Rml::Element* m_Element;

		friend class Canvas;
	};
}