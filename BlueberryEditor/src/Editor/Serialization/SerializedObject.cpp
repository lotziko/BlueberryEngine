#include "SerializedObject.h"

#include "Blueberry\Core\Base.h"
#include "SerializedProperty.h"
#include "Blueberry\Core\ClassDB.h"

#include "Editor\Misc\VariantHelper.h"
#include "Editor\Prefabs\PrefabManager.h"
#include "Editor\Prefabs\PrefabInstance.h"

namespace Blueberry
{
	#define EMPTY_ID UINT64_MAX

	ObjectUpdateEvent SerializedObject::s_ObjectUpdated = {};

	Object* ObjectUpdateEventArgs::GetObject() const
	{
		return m_Object;
	}

	SerializedObject::SerializedObject(Object* target)
	{
		m_ClassInfo = ClassDB::GetInfo(target->GetType());
		m_Targets.push_back(target);
		m_TargetPtrs.push_back(target);
		m_IsPrefabInstance.push_back(PrefabManager::IsOverridable(target));
		BuildTree();
	}

	SerializedObject::SerializedObject(List<Object*> targets)
	{
		m_ClassInfo = ClassDB::GetInfo(targets[0]->GetType());
		for (Object* target : targets)
		{
			m_Targets.push_back(target);
			m_TargetPtrs.push_back(target);
			m_IsPrefabInstance.push_back(PrefabManager::IsOverridable(target));
		}
		BuildTree();
	}

	SerializedProperty SerializedObject::FindProperty(const String& name)
	{
		for (auto& child : m_Nodes[0].children)
		{
			if (m_Nodes[child].fieldInfo->name == name)
			{
				return SerializedProperty(this, child);
			}
		}
		return {};
	}

	SerializedProperty SerializedObject::GetIterator()
	{
		SerializedProperty iterator = SerializedProperty(this, 0);
		iterator.m_Stack.push(std::make_pair(0, 0));
		return iterator;
	}

	Object* SerializedObject::GetTarget()
	{
		if (m_Targets.size() > 0)
		{
			return m_Targets[0];
		}
		return nullptr;
	}

	const List<Object*>& SerializedObject::GetTargets()
	{
		return m_Targets;
	}

	bool SerializedObject::IsValid()
	{
		for (size_t i = 0; i < m_Targets.size(); ++i)
		{
			if (!m_TargetPtrs[i].IsValid())
			{
				return false;
			}
		}
		return true;
	}

	void SerializedObject::Update()
	{
		// TODO rewrite into a single method that builds not existing nodes and reads existing
		//ReadTree();
		BuildTree();
	}

	bool SerializedObject::ApplyModifiedProperties()
	{
		if (m_Modifications.size() > 0)
		{
			for (PropertyModification modification : m_Modifications)
			{
				ApplyModification(modification);
			}
			m_Modifications.clear();
			// TODO use undo instead
			ObjectUpdateEventArgs args(m_Targets[0]);
			s_ObjectUpdated.Invoke(args);
			return true;
		}
		return false;
	}

	ObjectUpdateEvent& SerializedObject::GetObjectUpdated()
	{
		return s_ObjectUpdated;
	}

	void CalculateMixedMask(PropertyTreeNode* node)
	{
		memset(node->mixedMask, 0, sizeof(node->mixedMask));
		if (node->type == PropertyType::Value)
		{
			auto& values = node->values;
			auto& first = values[0];
			switch (node->bindingType)
			{
			case BindingType::Vector2:
			{
				Vector2 vectorFirst = std::get<Vector2>(first);
				for (size_t i = 1; i < values.size(); ++i)
				{
					Vector2 vectorValue = std::get<Vector2>(values[i]);
					if (vectorFirst.x != vectorValue.x)
					{
						node->mixedMask[0] = true;
					}
					if (vectorFirst.y != vectorValue.y)
					{
						node->mixedMask[1] = true;
					}
					break;
				}
			}
			break;
			case BindingType::Vector3:
			{
				Vector3 vectorFirst = std::get<Vector3>(first);
				for (size_t i = 1; i < values.size(); ++i)
				{
					Vector3 vectorValue = std::get<Vector3>(values[i]);
					if (vectorFirst.x != vectorValue.x)
					{
						node->mixedMask[0] = true;
					}
					if (vectorFirst.y != vectorValue.y)
					{
						node->mixedMask[1] = true;
					}
					if (vectorFirst.z != vectorValue.z)
					{
						node->mixedMask[2] = true;
					}
					break;
				}
			}
			break;
			case BindingType::Vector4:
			{
				Vector4 vectorFirst = std::get<Vector4>(first);
				for (size_t i = 1; i < values.size(); ++i)
				{
					Vector4 vectorValue = std::get<Vector4>(values[i]);
					if (vectorFirst.x != vectorValue.x)
					{
						node->mixedMask[0] = true;
					}
					if (vectorFirst.y != vectorValue.y)
					{
						node->mixedMask[1] = true;
					}
					if (vectorFirst.z != vectorValue.z)
					{
						node->mixedMask[2] = true;
					}
					if (vectorFirst.w != vectorValue.w)
					{
						node->mixedMask[3] = true;
					}
					break;
				}
			}
			break;
			case BindingType::Quaternion:
			{
				Quaternion quaternionFirst = std::get<Quaternion>(first);
				for (size_t i = 1; i < values.size(); ++i)
				{
					Quaternion quaternionValue = std::get<Quaternion>(values[i]);
					if (quaternionFirst.x != quaternionValue.x)
					{
						node->mixedMask[0] = true;
					}
					if (quaternionFirst.y != quaternionValue.y)
					{
						node->mixedMask[1] = true;
					}
					if (quaternionFirst.z != quaternionValue.z)
					{
						node->mixedMask[2] = true;
					}
					if (quaternionFirst.w != quaternionValue.w)
					{
						node->mixedMask[3] = true;
					}
					break;
				}
			}
			break;
			default:
				for (size_t i = 1; i < values.size(); ++i)
				{
					if (first == values[i])
					{
						continue;
					}
					node->mixedMask[0] = true;
					break;
				}
			}
		}
	}

	void SerializedObject::BuildTree()
	{
		m_Nodes.clear();
		size_t root = Allocate();
		PropertyTreeNode* rootNode = Get(root);
		rootNode->name = "Root";
		rootNode->parent = EMPTY_ID;
		rootNode->type = PropertyType::Root;
		rootNode->isVisible = false;

		const ClassInfo* classInfo = m_ClassInfo;
		List<void*> targets;
		for (Object* target : m_Targets)
		{
			targets.push_back(static_cast<void*>(target));
		}
		BuildProperties(root, classInfo, targets);
	}

	void SerializedObject::BuildProperties(size_t parent, const ClassInfo* classInfo, const List<void*>& targets)
	{
		for (size_t i = 0; i < classInfo->fields.size(); ++i)
		{
			const FieldInfo& fieldInfo = classInfo->fields[i];
			if (fieldInfo.options.visibility == VisibilityType::NonExposed)
			{
				continue;
			}
			size_t child = Allocate();
			PropertyTreeNode* childNode = Get(child);
			childNode->name = fieldInfo.name;
			childNode->parent = parent;
			childNode->index = i;
			childNode->fieldInfo = &fieldInfo;
			childNode->bindingType = fieldInfo.type;
			childNode->isVisible = fieldInfo.options.visibility == VisibilityType::Visible;
			if (fieldInfo.isList)
			{
				childNode->type = PropertyType::List;
				BuildList(child, fieldInfo, targets);
			}
			else if (fieldInfo.type == BindingType::Data)
			{
				childNode->type = PropertyType::Data;
				const ClassInfo* info = ClassDB::GetInfo(*fieldInfo.options.objectType);
				if (info == nullptr)
				{
					BB_ERROR("Class not exists.");
					return;
				}
				List<void*> newTargets = {};
				for (size_t j = 0; j < targets.size(); ++j)
				{
					newTargets.push_back(static_cast<char*>(targets[j]) + fieldInfo.offset);
				}
				BuildProperties(child, info, newTargets);
			}
			else
			{
				childNode->type = PropertyType::Value;
				for (size_t j = 0; j < targets.size(); ++j)
				{
					PropertyValue value = {};
					VariantHelper::ReadValue(fieldInfo.type, static_cast<char*>(targets[j]) + fieldInfo.offset, value);
					childNode->values.push_back(std::move(value));
					if (m_IsPrefabInstance[j] && PrefabManager::HasModification(m_Targets[j], GetNodePath(child)))
					{
						childNode->isOverriden = true;
					}
				}
			}
			CalculateMixedMask(Get(child));
			Get(parent)->children.push_back(child);
		}
	}

	void SerializedObject::BuildList(size_t parent, const FieldInfo& fieldInfo, const List<void*>& targets)
	{
		bool visible = fieldInfo.options.visibility == VisibilityType::Visible;

		size_t elementCount = UINT32_MAX;
		for (size_t i = 0; i < targets.size(); ++i)
		{
			PropertyTreeNode* parentNode = Get(parent);
			ListBase* list = fieldInfo.Get<ListBase>(targets[i]);
			size_t size = list->size_base();
			if (size < UINT32_MAX && size != elementCount)
			{
				parentNode->mixedMask[0] = true;
			}
			elementCount = std::min(elementCount, size);
		}

		BindingType childType = VariantHelper::GetChildType(fieldInfo.type);
		for (size_t i = 0; i < elementCount; ++i)
		{
			size_t listElement = Allocate();
			PropertyTreeNode* listElementNode = Get(listElement);
			listElementNode->name = "Element";
			listElementNode->parent = parent;
			listElementNode->index = i;
			listElementNode->fieldInfo = &fieldInfo;
			listElementNode->bindingType = childType;
			listElementNode->isVisible = visible;
			listElementNode->type = PropertyType::Value;
			Get(parent)->children.push_back(listElement);
		}
		
		if (fieldInfo.type == BindingType::DataList)
		{
			const ClassInfo* info = ClassDB::GetInfo(*fieldInfo.options.objectType);
			if (info == nullptr)
			{
				BB_ERROR("Class not exists.");
				return;
			}
			List<ListBase*> lists = {};
			for (size_t i = 0; i < targets.size(); ++i)
			{
				lists.push_back(fieldInfo.Get<ListBase>(targets[i]));
			}
			List<void*> newTargets = {};
			for (size_t i = 0; i < elementCount; ++i)
			{
				for (size_t j = 0; j < targets.size(); ++j)
				{
					newTargets.push_back(lists[j]->get_base(i));
				}
				BuildProperties(Get(parent)->children[i], info, newTargets);
				newTargets.clear();
			}
		}
		else
		{
			for (size_t i = 0; i < targets.size(); ++i)
			{
				ListBase* list = fieldInfo.Get<ListBase>(targets[i]);
				for (size_t j = 0; j < elementCount; ++j)
				{
					size_t listElement = Get(parent)->children[j];
					PropertyTreeNode* listElementNode = Get(listElement);
					PropertyValue value = {};
					VariantHelper::ReadValue(childType, list->get_base(j), value);
					listElementNode->values.push_back(std::move(value));
					if (m_IsPrefabInstance[i] && PrefabManager::HasModification(m_Targets[i], GetNodePath(listElement)))
					{
						listElementNode->isOverriden = true;
					}
				}
			}
		}
	}

	void SerializedObject::ReadTree()
	{
		const ClassInfo* classInfo = m_ClassInfo;
		List<void*> targets;
		for (Object* target : m_Targets)
		{
			targets.push_back(static_cast<void*>(target));
		}
		ReadProperties(0, classInfo, targets);
	}

	void SerializedObject::ReadProperties(size_t parent, const ClassInfo* classInfo, const List<void*>& targets)
	{
		PropertyTreeNode* parentNode = Get(parent);

		for (auto& child : parentNode->children)
		{
			PropertyTreeNode* childNode = Get(child);
			const FieldInfo* fieldInfo = childNode->fieldInfo;
			if (fieldInfo->isList)
			{
				ReadList(child, *fieldInfo, targets);
			}
			else if (fieldInfo->type == BindingType::Data)
			{
				const ClassInfo* info = ClassDB::GetInfo(*fieldInfo->options.objectType);
				if (info == nullptr)
				{
					BB_ERROR("Class not exists.");
					return;
				}
				List<void*> newTargets = {};
				for (size_t j = 0; j < targets.size(); ++j)
				{
					newTargets.push_back(static_cast<char*>(targets[j]) + fieldInfo->offset);
				}
				ReadProperties(child, info, newTargets);
			}
			else
			{
				for (size_t i = 0; i < targets.size(); ++i)
				{
					PropertyValue value = {};
					VariantHelper::ReadValue(fieldInfo->type, static_cast<char*>(targets[i]) + fieldInfo->offset, value);
					childNode->values[i] = std::move(value);
				}
			}
			CalculateMixedMask(childNode);
		}
	}

	void SerializedObject::ReadList(size_t parent, const FieldInfo& fieldInfo, const List<void*>& targets)
	{
		PropertyTreeNode* parentNode = Get(parent);
		size_t elementCount = parentNode->children.size();
		BindingType childType = VariantHelper::GetChildType(fieldInfo.type);
		if (fieldInfo.type == BindingType::DataList)
		{
			const ClassInfo* info = ClassDB::GetInfo(*fieldInfo.options.objectType);
			if (info == nullptr)
			{
				BB_ERROR("Class not exists.");
				return;
			}
			List<ListBase*> lists = {};
			for (size_t i = 0; i < targets.size(); ++i)
			{
				lists.push_back(fieldInfo.Get<ListBase>(targets[i]));
			}
			List<void*> newTargets = {};
			for (size_t i = 0; i < elementCount; ++i)
			{
				for (size_t j = 0; j < targets.size(); ++j)
				{
					newTargets.push_back(lists[j]->get_base(i));
				}
				ReadProperties(parentNode->children[i], info, newTargets);
				newTargets.clear();
			}
		}
		else
		{
			for (size_t i = 0; i < targets.size(); ++i)
			{
				ListBase* list = fieldInfo.Get<ListBase>(targets[i]);
				for (size_t j = 0; j < elementCount; ++j)
				{
					PropertyValue value = {};
					VariantHelper::ReadValue(childType, list->get_base(j), value);
					Get(parentNode->children[j])->values[i] = std::move(value);
				}
			}
		}
	}

	void SerializedObject::AddModifiedProperty(size_t id, const PropertyModificationType& type, const size_t& index1, const size_t& index2)
	{
		m_Modifications.push_back({ id, type, index1, index2 });
	}

	void SerializedObject::DeleteListElement(size_t id, size_t index)
	{
		m_Nodes[m_Nodes[id].children[index]].isDeleted = true;
		AddModifiedProperty(id, PropertyModificationType::Delete, index);
	}

	void SerializedObject::MoveListElement(size_t id, size_t fromIndex, size_t toIndex)
	{
		m_Nodes[id].children.move_element(fromIndex, toIndex);
		for (size_t i = 0; i < m_Nodes[id].children.size(); ++i)
		{
			m_Nodes[m_Nodes[id].children[i]].index = i;
		}
		AddModifiedProperty(id, PropertyModificationType::Move, fromIndex, toIndex);
	}

	void SerializedObject::ClearList(size_t id)
	{
		for (auto& childNode : m_Nodes[id].children)
		{
			m_Nodes[childNode].isDeleted = true;
		}
		AddModifiedProperty(id, PropertyModificationType::Clear);
	}

	size_t SerializedObject::Allocate()
	{
		size_t index = m_Nodes.size();
		m_Nodes.emplace_back();
		return index;
	}

	size_t SerializedObject::CreateChild(size_t parent)
	{
		PropertyTreeNode* parentNode = Get(parent);
		if (parentNode->type == PropertyType::List)
		{
			BindingType childType = VariantHelper::GetChildType(parentNode->bindingType);
			size_t child = Allocate();
			PropertyTreeNode* childNode = Get(child);
			childNode->name = "Element";
			childNode->parent = parent;
			childNode->fieldInfo = parentNode->fieldInfo;
			childNode->bindingType = childType;
			childNode->isVisible = parentNode->isVisible;
			childNode->type = PropertyType::Value;

			if (childType != BindingType::Data)
			{
				for (size_t i = 0; i < m_Targets.size(); ++i)
				{
					PropertyValue value = {};
					VariantHelper::GetDefaultValue(childType, value);
					childNode->values.push_back(std::move(value));
				}
			}
			return child;
		}
		return EMPTY_ID;
	}

	PropertyTreeNode* SerializedObject::Get(const size_t& id)
	{
		return &m_Nodes[id];
	}

	String SerializedObject::GetNodePath(size_t id)
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
				}
				else
				{
					result.append(".");
				}
			}
		}
		return result;
	}

	void SerializedObject::FindPath(PropertyTreeNode* node, List<void*>& result, size_t& offset)
	{
		size_t size = m_Targets.size();
		List<PropertyTreeNode*> path;
		PropertyTreeNode* currentNode = node;
		PropertyTreeNode* parentNode = nullptr;
		while (currentNode->type != PropertyType::Root)
		{
			path.push_back(currentNode);
			currentNode = Get(currentNode->parent);
		}

		for (size_t i = 0; i < size; ++i)
		{
			result.push_back(m_Targets[i]);
		}
		int pathSize = static_cast<int>(path.size());
		offset = 0;

		for (int i = pathSize - 1; i >= 0; --i)
		{
			currentNode = path[i];
			const FieldInfo* fieldInfo = currentNode->fieldInfo;
			// TODO check data
			if (parentNode != nullptr && parentNode->type == PropertyType::List)
			{
				offset = 0;
				for (size_t j = 0; j < size; ++j)
				{
					ListBase* list = parentNode->fieldInfo->Get<ListBase>(result[j]);
					result[j] = list->get_base(currentNode->index);
				}
			}
			else
			{
				offset += fieldInfo->offset;
			}
			parentNode = currentNode;
		}
	}

	void SerializedObject::ApplyModification(const PropertyModification& modification)
	{
		PropertyTreeNode* node = Get(modification.id);
		size_t offset = 0;
		switch (modification.type)
		{
		case PropertyModificationType::Value:
		{
			List<void*> targets;
			FindPath(node, targets, offset);
			const FieldInfo* fieldInfo = node->fieldInfo;
			MethodBind* callback = fieldInfo->options.updateCallback;
			for (size_t i = 0; i < node->values.size(); ++i)
			{
				if (PrefabManager::IsPartOfPrefabInstance(m_Targets[i]))
				{
					PrefabInstance* instance = PrefabManager::GetInstance(m_Targets[i]);
					if (instance->HasSource())
					{
						node->isOverriden = true;
						PrefabManager::AddModification(m_Targets[i], GetNodePath(modification.id), node->values[i]);
					}
				}
				VariantHelper::WriteValue(node->bindingType, static_cast<char*>(targets[i]) + offset, node->values[i]);
				if (callback != nullptr)
				{
					callback->Invoke(m_Targets[i]);
				}
			}
			CalculateMixedMask(node);
		}
		break;
		case PropertyModificationType::ClearOverride:
		{
			const FieldInfo* fieldInfo = node->fieldInfo;
			MethodBind* callback = fieldInfo->options.updateCallback;
			bool anyRemoved = false;
			String path = GetNodePath(modification.id);
			for (size_t i = 0; i < m_Targets.size(); ++i)
			{
				if (PrefabManager::IsPartOfPrefabInstance(m_Targets[i]))
				{
					PrefabInstance* instance = PrefabManager::GetInstance(m_Targets[i]);
					if (instance->HasSource())
					{
						PrefabManager::RemoveModification(m_Targets[i], path);
						if (callback != nullptr)
						{
							callback->Invoke(m_Targets[i]);
						}
						anyRemoved = true;
					}
				}
			}
			if (anyRemoved)
			{
				// TODO update only cleared part instead of updating entire tree
				node->isOverriden = false;
				Update();
			}
		}
		break;
		case PropertyModificationType::Insert:
		{
			List<void*> targets;
			FindPath(node, targets, offset);
			const FieldInfo* fieldInfo = node->fieldInfo;
			for (size_t i = 0; i < targets.size(); ++i)
			{
				ListBase* list = reinterpret_cast<ListBase*>(static_cast<char*>(targets[i]) + offset);
				list->insert_base(modification.index1);
			}
		}
		break;
		case PropertyModificationType::Delete:
		{
			List<void*> targets;
			FindPath(node, targets, offset);
			const FieldInfo* fieldInfo = node->fieldInfo;
			for (size_t i = 0; i < targets.size(); ++i)
			{
				ListBase* list = reinterpret_cast<ListBase*>(static_cast<char*>(targets[i]) + offset);
				list->erase_base(modification.index1);
			}
			node->children.erase(node->children.begin() + modification.index1);
		}
		break;
		case PropertyModificationType::Move:
		{
			List<void*> targets;
			FindPath(node, targets, offset);
			const FieldInfo* fieldInfo = node->fieldInfo;
			for (size_t i = 0; i < targets.size(); ++i)
			{
				ListBase* list = reinterpret_cast<ListBase*>(static_cast<char*>(targets[i]) + offset);
				list->move_element_base(modification.index1, modification.index2);
			}
		}
		break;
		case PropertyModificationType::Clear:
		{
			if (node->children.size() > 0)
			{
				List<void*> targets;
				FindPath(node, targets, offset);
				const FieldInfo* fieldInfo = node->fieldInfo;
				for (size_t i = 0; i < targets.size(); ++i)
				{
					ListBase* list = reinterpret_cast<ListBase*>(static_cast<char*>(targets[i]) + offset);
					list->clear_base();
				}
				node->children.clear();
			}
		}
		break;
		}
	}
}