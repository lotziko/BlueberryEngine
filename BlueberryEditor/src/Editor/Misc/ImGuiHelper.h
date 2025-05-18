#pragma once
#include "Blueberry\Core\ObjectDB.h"

namespace Blueberry
{
	template<class ObjectType>
	class ObjectPtr;
}

namespace ImGui
{
	struct EditorStyle
	{
		float ProjectBottomPanelSize;
		float ProjectCellSize;
		float ProjectSpaceBetweenCells;
		float ProjectCellIconPadding;
		float ProjectExpandIconSize;
		float ProjectFolderIconSize;
	};

	struct EditorContext
	{
		EditorStyle Style;
	};

	extern EditorContext* GEditor;

	void CreateEditorContext();
	EditorStyle& GetEditorStyle();

	bool DragVector2(const char* label, Blueberry::Vector2* v);
	bool DragVector3(const char* label, Blueberry::Vector3* v);
	bool EnumEdit(const char* label, int* v, const Blueberry::List<Blueberry::String>* names);
	bool BoolEdit(const char* label, bool* v);
	bool IntEdit(const char* label, int* v);
	bool FloatEdit(const char* label, float* v, float min = 0, float max = 0);
	bool ColorEdit(const char* label, Blueberry::Color* v);
	bool ObjectEdit(const char* label, Blueberry::Object** v, const size_t& type);
	bool ObjectEdit(const char* label, Blueberry::ObjectPtr<Blueberry::Object>* v, const size_t& type);
	bool ObjectArrayEdit(const char* label, Blueberry::List<Blueberry::ObjectPtr<Blueberry::Object>>* v, const size_t& type);
	
	bool SearchInputText(const char* hint, std::string* text);

	void HorizontalSplitter(const char* strId, float* size, float minSize);

	void ApplyEditorDarkTheme();
	void LoadDefaultEditorFonts();
}