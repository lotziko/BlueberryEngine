#pragma once
#include "Blueberry\Core\ObjectDB.h"

namespace Blueberry
{
	template<class ObjectType>
	class ObjectPtr;
}

namespace ImGui
{
	bool DragVector3(const char* label, Blueberry::Vector3* v);
	bool ColorEdit(const char* label, Blueberry::Color* v);
	bool ObjectEdit(const char* label, Blueberry::ObjectPtr<Blueberry::Object>* v, const std::size_t& type);

	void ApplyEditorDarkTheme();
	void LoadDefaultEditorFonts();
}