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
		m_Targets.push_back(target);
		BuildTree();
	}

	SerializedObject::SerializedObject(List<Object*> targets)
	{
		m_ClassInfo = &ClassDB::GetInfo(targets[0]->GetType());
		for (Object* target : targets)
		{
			m_Targets.push_back(target);
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

	void SerializedObject::Update()
	{
		ReadTree();
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
			return true;
		}
		return false;
	}

	void GetDefaultValue(const BindingType& type, PropertyValue& value)
	{
		switch (type)
		{
		case BindingType::Bool:
			value = false;
			break;
		case BindingType::Int:
			value = 0;
			break;
		case BindingType::Float:
			value = 0.0f;
			break;
		case BindingType::Enum:
			value = 0;
			break;
		case BindingType::String:
			value = "";
			break;
		case BindingType::Vector2:
			value = Vector2::Zero;
			break;
		case BindingType::Vector3:
			value = Vector3::Zero;
			break;
		case BindingType::Vector4:
			value = Vector4::Zero;
			break;
		case BindingType::Quaternion:
			value = Quaternion::Identity;
			break;
		case BindingType::Color:
			value = Color(0, 0, 0, 0);
			break;
		case BindingType::ObjectPtr:
			value = ObjectPtr<Object>();
			break;
		}
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
		case BindingType::IntList:
			return BindingType::Int;
		case BindingType::FloatList:
			return BindingType::Float;
		case BindingType::StringList:
			return BindingType::String;
		case BindingType::Vector2List:
			return BindingType::Vector2;
		case BindingType::Vector3List:
			return BindingType::Vector3;
		case BindingType::Vector4List:
			return BindingType::Vector4;
		case BindingType::ObjectPtrList:
			return BindingType::ObjectPtr;
		case BindingType::DataList:
			return BindingType::Data;
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
		m_Root = std::make_shared<PropertyTreeNode>();
		m_Root->name = "Root";
		m_Root->type = PropertyType::Root;
		m_Root->isVisible = false;

		const ClassInfo* classInfo = m_ClassInfo;
		List<void*> targets;
		for (Object* target : m_Targets)
		{
			targets.push_back(static_cast<void*>(target));
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
					childNode->values.push_back(std::move(value));
				}
			}
			CalculateMixedMask(childNode.get());
			parentShared->children.push_back(std::move(childNode));
		}
	}

	void SerializedObject::BuildList(std::weak_ptr<PropertyTreeNode> parent, const FieldInfo& fieldInfo, const List<void*>& targets)
	{
		std::shared_ptr<PropertyTreeNode> parentShared = parent.lock();
		bool visible = fieldInfo.options.visibility == VisibilityType::Visible;

		size_t elementCount = UINT32_MAX;
		for (size_t i = 0; i < targets.size(); ++i)
		{
			Variant variant(targets[i], fieldInfo.offset);
			ListBase* list = variant.Get<ListBase>();
			size_t size = list->size_base();
			if (size < UINT32_MAX && size != elementCount)
			{
				parentShared->mixedMask[0] = true;
			}
			elementCount = std::min(elementCount, size);
		}

		BindingType childType = GetChildType(fieldInfo.type);
		for (size_t i = 0; i < elementCount; ++i)
		{
			std::shared_ptr<PropertyTreeNode> listNode = std::make_shared<PropertyTreeNode>();
			listNode->name = "Element";
			listNode->parent = parent;
			listNode->index = i;
			listNode->fieldInfo = &fieldInfo;
			listNode->bindingType = childType;
			listNode->isVisible = visible;
			listNode->type = PropertyType::Value;
			parentShared->children.push_back(std::move(listNode));
		}

		if (fieldInfo.type == BindingType::DataList)
		{
			const ClassInfo& info = ClassDB::GetInfo(fieldInfo.options.objectType);
			List<ListBase*> lists = {};
			for (size_t i = 0; i < targets.size(); ++i)
			{
				Variant variant(targets[i], fieldInfo.offset);
				lists.push_back(variant.Get<ListBase>());
			}
			List<void*> newTargets = {};
			for (size_t i = 0; i < elementCount; ++i)
			{
				for (size_t j = 0; j < targets.size(); ++j)
				{
					newTargets.push_back(lists[j]->get_base(i));
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
				for (size_t j = 0; j < elementCount; ++j)
				{
					PropertyValue value = {};
					variant = Variant(list->get_base(j), 0);
					ReadValue(childType, variant, value);
					parentShared->children[j]->values.push_back(std::move(value));
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
		ReadProperties(m_Root, classInfo, targets);
	}

	void SerializedObject::ReadProperties(std::weak_ptr<PropertyTreeNode> parent, const ClassInfo* classInfo, const List<void*>& targets)
	{
		std::shared_ptr<PropertyTreeNode> parentShared = parent.lock();

		for (auto& childNode : parentShared->children)
		{
			const FieldInfo* fieldInfo = childNode->fieldInfo;
			if (fieldInfo->isList)
			{
				ReadList(childNode, *fieldInfo, targets);
			}
			else
			{
				for (size_t i = 0; i < targets.size(); ++i)
				{
					Variant variant(targets[i], fieldInfo->offset);
					PropertyValue value = {};
					ReadValue(fieldInfo->type, variant, value);
					childNode->values[i] = std::move(value);
				}
			}
			CalculateMixedMask(childNode.get());
		}
	}

	void SerializedObject::ReadList(std::weak_ptr<PropertyTreeNode> parent, const FieldInfo& fieldInfo, const List<void*>& targets)
	{
		std::shared_ptr<PropertyTreeNode> parentShared = parent.lock();
		size_t elementCount = parentShared->children.size();
		BindingType childType = GetChildType(fieldInfo.type);
		if (fieldInfo.type == BindingType::DataList)
		{
			const ClassInfo& info = ClassDB::GetInfo(fieldInfo.options.objectType);
			List<ListBase*> lists = {};
			for (size_t i = 0; i < targets.size(); ++i)
			{
				Variant variant(targets[i], fieldInfo.offset);
				lists.push_back(variant.Get<ListBase>());
			}
			List<void*> newTargets = {};
			for (size_t i = 0; i < elementCount; ++i)
			{
				for (size_t j = 0; j < targets.size(); ++j)
				{
					newTargets.push_back(lists[j]->get_base(i));
				}
				ReadProperties(parentShared->children[i], &info, newTargets);
				newTargets.clear();
			}
		}
		else
		{
			for (size_t i = 0; i < targets.size(); ++i)
			{
				Variant variant(targets[i], fieldInfo.offset);
				ListBase* list = variant.Get<ListBase>();
				for (size_t j = 0; j < elementCount; ++j)
				{
					PropertyValue value = {};
					variant = Variant(list->get_base(j), 0);
					ReadValue(childType, variant, value);
					parentShared->children[j]->values[i] = std::move(value);
				}
			}
		}
	}

	void SerializedObject::AddModifiedProperty(PropertyTreeNode* node, const PropertyModificationType& type, const size_t& index1, const size_t& index2)
	{
		m_Modifications.push_back({ node, type, index1, index2 });
	}

	std::shared_ptr<PropertyTreeNode> SerializedObject::CreateChild(PropertyTreeNode* parent)
	{
		if (parent->type == PropertyType::List)
		{
			std::shared_ptr<PropertyTreeNode> parentShared;
			for (auto& child : parent->parent.lock()->children)
			{
				if (child.get() == parent)
				{
					parentShared = child;
					break;
				}
			}

			BindingType childType = GetChildType(parent->bindingType);
			std::shared_ptr<PropertyTreeNode> listNode = std::make_shared<PropertyTreeNode>();
			listNode->name = "Element";
			listNode->parent = parentShared;
			listNode->fieldInfo = parentShared->fieldInfo;
			listNode->bindingType = childType;
			listNode->isVisible = parentShared->isVisible;
			listNode->type = PropertyType::Value;

			if (childType != BindingType::Data)
			{
				for (size_t i = 0; i < m_Targets.size(); ++i)
				{
					PropertyValue value = {};
					GetDefaultValue(childType, value);
					listNode->values.push_back(std::move(value));
				}
			}

			return listNode;
		}
		return nullptr;
	}

	void SerializedObject::FindPath(PropertyTreeNode* node, List<char*>& result, size_t& offset)
	{
		size_t size = m_Targets.size();
		List<PropertyTreeNode*> path;
		PropertyTreeNode* currentNode = node;
		PropertyTreeNode* parentNode = nullptr;
		while (currentNode->type != PropertyType::Root)
		{
			path.push_back(currentNode);
			currentNode = currentNode->parent.lock().get();
		}

		for (size_t i = 0; i < size; ++i)
		{
			result.push_back(reinterpret_cast<char*>(m_Targets[i]));
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
					Variant variant(result[j], parentNode->fieldInfo->offset);
					ListBase* list = variant.Get<ListBase>();
					result[j] = reinterpret_cast<char*>(list->get_base(currentNode->index));
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
		PropertyTreeNode* node = modification.node;
		size_t offset = 0;
		switch (modification.type)
		{
		case PropertyModificationType::Value:
		{
			List<char*> targets;
			FindPath(node, targets, offset);
			size_t size = node->values.size();
			const FieldInfo* fieldInfo = node->fieldInfo;
			MethodBind* callback = fieldInfo->options.updateCallback;
			for (size_t i = 0; i < size; ++i)
			{
				Variant variant(targets[i], offset);
				WriteValue(node->bindingType, variant, node->values[i]);
				if (callback != nullptr)
				{
					callback->Invoke(m_Targets[i]);
				}
			}
			CalculateMixedMask(node);
		}
		break;
		case PropertyModificationType::Insert:
		{
			List<char*> targets;
			FindPath(node, targets, offset);
			const FieldInfo* fieldInfo = node->fieldInfo;
			for (size_t i = 0; i < targets.size(); ++i)
			{
				Variant variant(targets[i], offset);
				ListBase* list = variant.Get<ListBase>();
				list->insert_base(modification.index1);
			}
		}
		break;
		case PropertyModificationType::Delete:
		{
			List<char*> targets;
			FindPath(node, targets, offset);
			const FieldInfo* fieldInfo = node->fieldInfo;
			for (size_t i = 0; i < targets.size(); ++i)
			{
				Variant variant(targets[i], offset);
				ListBase* list = variant.Get<ListBase>();
				list->erase_base(modification.index1);
			}
			node->children.erase(node->children.begin() + modification.index1);
		}
		break;
		case PropertyModificationType::Move:
		{
			List<char*> targets;
			FindPath(node, targets, offset);
			const FieldInfo* fieldInfo = node->fieldInfo;
			for (size_t i = 0; i < targets.size(); ++i)
			{
				Variant variant(targets[i], offset);
				ListBase* list = variant.Get<ListBase>();
				list->move_element_base(modification.index1, modification.index2);
			}
		}
		break;
		case PropertyModificationType::Clear:
		{
			if (node->children.size() > 0)
			{
				List<char*> targets;
				FindPath(node, targets, offset);
				const FieldInfo* fieldInfo = node->fieldInfo;
				for (size_t i = 0; i < targets.size(); ++i)
				{
					Variant variant(targets[i], offset);
					ListBase* list = variant.Get<ListBase>();
					list->clear_base();
				}
				node->children.clear();
			}
		}
		break;
		}
	}
}