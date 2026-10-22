#include "ObjectSnapshot.h"

#include "Blueberry\Core\ClassDB.h"
#include "Editor\Misc\VariantHelper.h"

namespace Blueberry
{
	#define EMPTY_ID UINT64_MAX

	ObjectSnapshot::ObjectSnapshot(Object* object)
	{
		size_t root = Allocate();
		PropertyTreeNode* rootNode = Get(root);
		rootNode->type = PropertyType::Root;
		rootNode->parent = EMPTY_ID;

		const ClassInfo* classInfo = ClassDB::GetInfo(object->GetType());
		BuildProperties(root, classInfo, object);
	}

	void ObjectSnapshot::Compare(Object* object, List<FieldValueChange>& differences)
	{
		const ClassInfo* classInfo = ClassDB::GetInfo(object->GetType());
		size_t index = 0;
		CompareProperties(index, classInfo, object, differences);
	}

	void ObjectSnapshot::BuildProperties(size_t parent, const ClassInfo* classInfo, void* target)
	{
		for (size_t i = 0; i < classInfo->fields.size(); ++i)
		{
			const FieldInfo* fieldInfo = &classInfo->fields[i];
			size_t child = Allocate();
			PropertyTreeNode* childNode = Get(child);
			childNode->parent = parent;
			childNode->fieldInfo = fieldInfo;
			Get(parent)->children.push_back(child);

			if (fieldInfo->isList)
			{
				childNode->type = PropertyType::List;
				childNode->name = fieldInfo->name;
				BuildList(child, fieldInfo, target);
			}
			else if (fieldInfo->type == BindingType::Data)
			{
				childNode->type = PropertyType::Data;
				childNode->name = fieldInfo->name;
				const ClassInfo* info = ClassDB::GetInfo(*fieldInfo->options.objectType);
				if (info == nullptr)
				{
					BB_ERROR("Class not exists.");
					return;
				}
				BuildProperties(child, info, static_cast<char*>(target) + fieldInfo->offset);
			}
			else
			{
				childNode->type = PropertyType::Value;
				childNode->name = fieldInfo->name;
				VariantHelper::ReadValue(fieldInfo->type, static_cast<char*>(target) + fieldInfo->offset, childNode->value);
			}
		}
	}

	void ObjectSnapshot::BuildList(size_t parent, const FieldInfo* fieldInfo, void* target)
	{
		ListBase* list = fieldInfo->Get<ListBase>(target);
		size_t size = list->size_base();
		Get(parent)->childCount = size;

		if (fieldInfo->type == BindingType::DataList)
		{
			const ClassInfo* info = ClassDB::GetInfo(*fieldInfo->options.objectType);
			if (info == nullptr)
			{
				BB_ERROR("Class not exists.");
				return;
			}
			for (size_t i = 0; i < size; ++i)
			{
				size_t listElement = Allocate();
				PropertyTreeNode* listElementNode = Get(listElement);
				listElementNode->type = PropertyType::Data;
				listElementNode->index = i;
				listElementNode->parent = parent;
				Get(parent)->children.push_back(listElement);
				BuildProperties(listElement, info, list->get_base(i));
			}
		}
		else
		{
			for (size_t i = 0; i < size; ++i)
			{
				size_t listElement = Allocate();
				PropertyTreeNode* listElementNode = Get(listElement);
				listElementNode->type = PropertyType::Value;
				listElementNode->index = i;
				listElementNode->parent = parent;
				Get(parent)->children.push_back(listElement);
				VariantHelper::ReadValue(fieldInfo->type, list->get_base(i), listElementNode->value);
			}
		}
	}

	void ObjectSnapshot::CompareProperties(size_t parent, const ClassInfo* classInfo, void* target, List<FieldValueChange>& differences)
	{
		PropertyTreeNode* parentNode = Get(parent);
		for (size_t child : parentNode->children)
		{
			const FieldInfo* fieldInfo = Get(child)->fieldInfo;
			if (fieldInfo->isList)
			{
				CompareList(child, fieldInfo, target, differences);
			}
			else if (fieldInfo->type == BindingType::Data)
			{
				const ClassInfo* info = ClassDB::GetInfo(*fieldInfo->options.objectType);
				CompareProperties(child, info, static_cast<char*>(target) + fieldInfo->offset, differences);
			}
			else
			{
				Variant value;
				VariantHelper::ReadValue(fieldInfo->type, static_cast<char*>(target) + fieldInfo->offset, value);
				PropertyTreeNode* childNode = Get(child);
				if (!(value == childNode->value))
				{
					differences.push_back({ GetNodePath(child), childNode->value, value });	// Should build path, not use name
				}
			}
		}
	}

	void ObjectSnapshot::CompareList(size_t parent, const FieldInfo* fieldInfo, void* target, List<FieldValueChange>& differences)
	{
		PropertyTreeNode* parentNode = Get(parent);
		ListBase* list = fieldInfo->Get<ListBase>(target);
		size_t size = list->size_base();
		if (parentNode->childCount == size)
		{
			if (fieldInfo->type == BindingType::DataList)
			{
				const ClassInfo* info = ClassDB::GetInfo(*fieldInfo->options.objectType);
				if (info == nullptr)
				{
					BB_ERROR("Class not exists.");
					return;
				}
				for (size_t i = 0; i < size; ++i)
				{
					CompareProperties(parentNode->children[i], info, list->get_base(i), differences);
				}
			}
			else
			{
				for (size_t i = 0; i < size; ++i)
				{
					Variant value;
					VariantHelper::ReadValue(fieldInfo->type, list->get_base(i), value);
					size_t child = parentNode->children[i];
					PropertyTreeNode* childNode = Get(child);
					if (!(value == childNode->value))
					{
						differences.push_back({ GetNodePath(child), childNode->value, value });
					}
				}
			}
		}
		else
		{
			// TODO
		}
	}

	size_t ObjectSnapshot::Allocate()
	{
		size_t index = m_Nodes.size();
		m_Nodes.emplace_back();
		return index;
	}

	ObjectSnapshot::PropertyTreeNode* ObjectSnapshot::Get(size_t id)
	{
		return &m_Nodes[id];
	}

	String ObjectSnapshot::GetNodePath(size_t id)
	{
		List<size_t> path;
		while (id != EMPTY_ID)
		{
			path.push_back(id);
			id = Get(id)->parent;
		}
		String result = "";
		for (size_t i = path.size() - 1; i--;)	// - 1 avoids adding root name
		{
			PropertyTreeNode* pathNode = Get(path[i]);
			result.append(pathNode->name);
			if (i > 0)
			{
				if (pathNode->type == PropertyType::List)
				{
					result.append("[");
					result.append(std::to_string(Get(path[i - 1])->index));
					result.append("]");
					--i;
					if (i > 0)
					{
						result.append(".");
					}
				}
				else
				{
					result.append(".");
				}
			}
		}
		return result;
	}
}