#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\Variant.h"

namespace Blueberry
{
	class Object;
	struct ClassInfo;
	struct FieldInfo;

	struct FieldValueChange
	{
		String path;
		Variant before;
		Variant after;
	};

	class ObjectSnapshot
	{
	public:
		BB_OVERRIDE_NEW_DELETE

		ObjectSnapshot() = default;
		ObjectSnapshot(Object* object);

		void Compare(Object* object, List<FieldValueChange>& differences);

	private:
		enum class PropertyType
		{
			Root,
			Value,
			List,
			Data
		};

		struct PropertyTreeNode
		{
			PropertyType type;
			String name;
			const FieldInfo* fieldInfo;
			size_t index;
			Variant value;
			size_t childCount;

			size_t parent;
			List<size_t> children;
		};

		void BuildProperties(size_t parent, const ClassInfo* classInfo, void* target);
		void BuildList(size_t parent, const FieldInfo* fieldInfo, void* target);

		void CompareProperties(size_t parent, const ClassInfo* classInfo, void* target, List<FieldValueChange>& differences);
		void CompareList(size_t parent, const FieldInfo* fieldInfo, void* target, List<FieldValueChange>& differences);
		
		size_t Allocate();
		PropertyTreeNode* Get(size_t id);
		String GetNodePath(size_t id);

	private:
		List<PropertyTreeNode> m_Nodes = {};
	};
}