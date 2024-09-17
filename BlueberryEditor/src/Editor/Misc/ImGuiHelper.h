#pragma once
#include "Blueberry\Core\ObjectDB.h"

namespace Blueberry
{
	template<class ObjectType>
	class ObjectPtr;
}

namespace ImGui
{
	bool DragVector2(const char* label, Blueberry::Vector2* v);
	bool DragVector3(const char* label, Blueberry::Vector3* v);
	bool EnumEdit(const char* label, int* v, const std::vector<std::string>* names);
	bool BoolEdit(const char* label, bool* v);
	bool IntEdit(const char* label, int* v);
	bool FloatEdit(const char* label, float* v);
	bool ColorEdit(const char* label, Blueberry::Color* v);
	bool ObjectEdit(const char* label, Blueberry::Object** v, const std::size_t& type);
	bool ObjectEdit(const char* label, Blueberry::ObjectPtr<Blueberry::Object>* v, const std::size_t& type);
	bool ObjectArrayEdit(const char* label, std::vector<Blueberry::ObjectPtr<Blueberry::Object>>* v, const std::size_t& type);

	void ApplyEditorDarkTheme();
	void LoadDefaultEditorFonts();
}