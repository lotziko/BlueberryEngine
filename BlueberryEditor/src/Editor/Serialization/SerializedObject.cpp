#include "SerializedObject.h"

#include "Blueberry\Core\Base.h"
#include "SerializedProperty.h"
#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Core\Variant.h"

namespace Blueberry
{
	SerializedObject::SerializedObject(Object* target)
	{
		m_ClassInfo = &ClassDB::GetInfo(target->GetType());
		m_Targets.emplace_back(target);
		BuildTree();
	}

	SerializedObject::SerializedObject(List<Object*> targets)
	{
		m_ClassInfo = &ClassDB::GetInfo(targets[0]->GetType());
		for (Object* target : targets)
		{
			m_Targets.emplace_back(target);
		}
		BuildTree();
	}

	SerializedProperty SerializedObject::FindProperty(const String& name)
	{
		for (auto& child : m_Root->children)
		{
			if (child->fieldInfo->name == name)
			{
				return SerializedProperty(this, child.get());
			}
		}
	}

	SerializedProperty SerializedObject::GetIterator()
	{
		SerializedProperty iterator = SerializedProperty(this, m_Root.get());
		iterator.m_Stack.push(std::make_pair(iterator.m_TreeNode, 0));
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

	bool SerializedObject::ApplyModifiedProperties()
	{
		if (m_ModifiedNodes.size() > 0)
		{
			for (PropertyTreeNode* node : m_ModifiedNodes)
			{
				ApplyValues(node);
			}
			m_ModifiedNodes.clear();
			return true;
		}
		return false;
	}

	void ReadValue(const BindingType& type, Variant& variant, PropertyValue& value)
	{
		switch (type)
		{
		case BindingType::Bool:
			value = *variant.Get<bool>();
			break;
		case BindingType::Int:
			value = *variant.Get<int>();
			break;
		case BindingType::Float:
			value = *variant.Get<float>();
			break;
		case BindingType::Enum:
			value = *variant.Get<int>();
			break;
		case BindingType::String:
			value = *variant.Get<String>();
			break;
		case BindingType::Vector2:
			value = *variant.Get<Vector2>();
			break;
		case BindingType::Vector3:
			value = *variant.Get<Vector3>();
			break;
		case BindingType::Vector4:
			value = *variant.Get<Vector4>();
			break;
		case BindingType::Quaternion:
			value = *variant.Get<Quaternion>();
			break;
		case BindingType::Color:
			value = *variant.Get<Color>();
			break;
		case BindingType::ObjectPtr:
			value = *variant.Get<ObjectPtr<Object>>();
			break;
		}
	}

	void WriteValue(const BindingType& type, Variant& variant, PropertyValue& value)
	{
		switch (type)
		{
		case BindingType::Bool:
			*variant.Get<bool>() = std::get<bool>(value);
			break;
		case BindingType::Int:
			*variant.Get<int>() = std::get<int>(value);
			break;
		case BindingType::Float:
			*variant.Get<float>() = std::get<float>(value);
			break;
		case BindingType::Enum:
			*variant.Get<int>() = std::get<int>(value);
			break;
		case BindingType::String:
			*variant.Get<String>() = std::get<String>(value);
			break;
		case BindingType::Vector2:
			*variant.Get<Vector2>() = std::get<Vector2>(value);
			break;
		case BindingType::Vector3:
			*variant.Get<Vector3>() = std::get<Vector3>(value);
			break;
		case BindingType::Vector4:
			*variant.Get<Vector4>() = std::get<Vector4>(value);
			break;
		case BindingType::Quaternion:
			*variant.Get<Quaternion>() = std::get<Quaternion>(value);
			break;
		case BindingType::Color:
			*variant.Get<Color>() = std::get<Color>(value);
			break;
		case BindingType::ObjectPtr:
			*variant.Get<ObjectPtr<Object>>() = std::get<ObjectPtr<Object>>(value);
			break;
		}
	}

	BindingType GetChildType(const BindingType& type)
	{
		switch (type)
		{
		case BindingType::ObjectPtrList:
			return BindingType::ObjectPtr;
		default:
			return BindingType::None;
		}
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
				for (uint32_t i = 1; i < values.size(); ++i)
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
				for (uint32_t i = 1; i < values.size(); ++i)
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
				for (uint32_t i = 1; i < values.size(); ++i)
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
				for (uint32_t i = 1; i < values.size(); ++i)
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
				for (uint32_t i = 1; i < values.size(); ++i)
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
		m_Root = std::make_shared<PropertyTreeNode>();
		m_Root->name = "Root";
		m_Root->type = PropertyType::Root;
		m_Root->isVisible = false;

		const ClassInfo* classInfo = m_ClassInfo;
		List<void*> targets;
		for (Object* target : m_Targets)
		{
			targets.emplace_back(static_cast<void*>(target));
		}
		BuildProperties(m_Root, classInfo, targets);
	}

	void SerializedObject::BuildProperties(std::weak_ptr<PropertyTreeNode> parent, const ClassInfo* classInfo, const List<void*>& targets)
	{
		std::shared_ptr<PropertyTreeNode> parentShared = parent.lock();

		for (size_t i = 0; i < classInfo->fields.size(); ++i)
		{
			const FieldInfo& fieldInfo = classInfo->fields[i];
			if (fieldInfo.options.visibility == VisibilityType::NonExposed)
			{
				continue;
			}
			std::shared_ptr<PropertyTreeNode> childNode = std::make_shared<PropertyTreeNode>();
			childNode->name = fieldInfo.name;
			childNode->parent = parent;
			childNode->index = i;
			childNode->fieldInfo = &fieldInfo;
			childNode->bindingType = fieldInfo.type;
			childNode->isVisible = fieldInfo.options.visibility == VisibilityType::Visible;
			if (fieldInfo.isList)
			{
				childNode->type = PropertyType::List;
				BuildList(childNode, fieldInfo, targets);
			}
			else
			{
				childNode->type = PropertyType::Value;
				for (size_t j = 0; j < targets.size(); ++j)
				{
					Variant variant(targets[j], fieldInfo.offset);
					PropertyValue value = {};
					ReadValue(fieldInfo.type, variant, value);
					childNode->values.emplace_back(std::move(value));
				}
			}
			CalculateMixedMask(childNode.get());
			parentShared->children.emplace_back(std::move(childNode));
		}
	}

	void SerializedObject::BuildList(std::weak_ptr<PropertyTreeNode> parent, const FieldInfo& fieldInfo, const List<void*>& targets)
	{
		std::shared_ptr<PropertyTreeNode> parentShared = parent.lock();
		bool visible = fieldInfo.options.visibility == VisibilityType::Visible;

		uint32_t elementCount = UINT32_MAX;
		for (size_t i = 0; i < targets.size(); ++i)
		{
			Variant variant(targets[i], fieldInfo.offset);
			ListBase* list = variant.Get<ListBase>();
			uint32_t size = static_cast<uint32_t>(list->size_base());
			if (size < UINT32_MAX && size != elementCount)
			{
				parentShared->mixedMask[0] = true;
			}
			elementCount = std::min(elementCount, size);
		}

		BindingType childType = GetChildType(fieldInfo.type);
		for (uint32_t i = 0; i < elementCount; ++i)
		{
			std::shared_ptr<PropertyTreeNode> listNode = std::make_shared<PropertyTreeNode>();
			String name = "Element ";
			name.append(std::to_string(i));
			listNode->name = name;
			listNode->parent = parent;
			listNode->index = i;
			listNode->fieldInfo = &fieldInfo;
			listNode->bindingType = childType;
			listNode->isVisible = visible;
			listNode->type = PropertyType::Value;
			parentShared->children.emplace_back(std::move(listNode));
		}

		if (fieldInfo.type == BindingType::DataList)
		{
			const ClassInfo& info = ClassDB::GetInfo(fieldInfo.options.objectType);
			List<ListBase*> lists = {};
			for (size_t i = 0; i < targets.size(); ++i)
			{
				Variant variant(targets[i], fieldInfo.offset);
				lists.emplace_back(variant.Get<ListBase>());
			}
			List<void*> newTargets = {};
			for (uint32_t i = 0; i < elementCount; ++i)
			{
				for (size_t j = 0; j < targets.size(); ++j)
				{
					newTargets.emplace_back(lists[j]->get_base(i));
				}
				BuildProperties(parentShared->children[i], &info, newTargets);
				newTargets.clear();
			}
		}
		else
		{
			for (size_t i = 0; i < targets.size(); ++i)
			{
				Variant variant(targets[i], fieldInfo.offset);
				ListBase* list = variant.Get<ListBase>();
				for (uint32_t j = 0; j < elementCount; ++j)
				{
					PropertyValue value = {};
					variant = Variant(list->get_base(j), 0);
					ReadValue(childType, variant, value);
					parentShared->children[j]->values.emplace_back(std::move(value));
				}
			}
		}
	}

	void SerializedObject::AddModifiedProperty(SerializedProperty* property)
	{
		m_ModifiedNodes.emplace_back(std::move(property->m_TreeNode));
	}

	void SerializedObject::ApplyValues(PropertyTreeNode* node)
	{
		uint32_t size = static_cast<uint32_t>(node->values.size());
		List<PropertyTreeNode*> path;
		PropertyTreeNode* currentNode = node;
		PropertyTreeNode* parentNode = nullptr;
		while (currentNode->type != PropertyType::Root)
		{
			path.emplace_back(currentNode);
			currentNode = currentNode->parent.lock().get();
		}
		List<char*> targets;
		for (uint32_t i = 0; i < size; ++i)
		{
			targets.emplace_back(reinterpret_cast<char*>(m_Targets[i]));
		}
		int pathSize = static_cast<int>(path.size());
		uint32_t offset = 0;
		for (int i = pathSize - 1; i >= 0; --i)
		{
			currentNode = path[i];
			const FieldInfo* fieldInfo = currentNode->fieldInfo;
			// TODO check data
			if (parentNode != nullptr && parentNode->type == PropertyType::List)
			{
				offset = 0;
				for (uint32_t j = 0; j < size; ++j)
				{
					Variant variant(targets[j], parentNode->fieldInfo->offset);
					ListBase* list = variant.Get<ListBase>();
					targets[j] = reinterpret_cast<char*>(list->get_base(currentNode->index));
				}
			}
			else
			{
				offset += fieldInfo->offset;
			}
			if (currentNode->type == PropertyType::Value)
			{
				MethodBind* callback = fieldInfo->options.updateCallback;
				for (uint32_t j = 0; j < size; ++j)
				{
					Variant variant(targets[j], offset);
					WriteValue(currentNode->bindingType, variant, currentNode->values[j]);
					if (callback != nullptr)
					{
						callback->Invoke(m_Targets[j]);
					}
				}
			}
			parentNode = currentNode;
		}
		CalculateMixedMask(node);
	}
}