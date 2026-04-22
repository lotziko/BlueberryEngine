#include "Blueberry\UI\UIElement.h"

#include <RmlUi\Core\Element.h>

namespace Blueberry
{
	UIElement UIElement::GetElementById(const char* name)
	{
		UIElement element = {};
		element.m_Element = m_Element->GetElementById(name);
		return element;
	}

	void UIElement::SetProperty(const char* name, const char* value)
	{
		m_Element->SetProperty(name, value);
	}
}