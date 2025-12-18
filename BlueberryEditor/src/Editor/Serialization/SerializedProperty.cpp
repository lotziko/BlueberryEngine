#include "SerializedProperty.h"

#include "SerializedObject.h"
#include "Blueberry\Core\Object.h"
#include "Blueberry\Core\ClassDB.h"

namespace Blueberry
{
	size_t SerializedProperty::GetId()
	{
		return reinterpret_cast<size_t>(m_TreeNode);
	}

	const String& SerializedProperty::GetName()
	{
		return m_TreeNode->name;
	}

	const BindingType SerializedProperty::GetType()
	{
		return m_TreeNode->bindingType;
	}

	const bool SerializedProperty::IsMixedValue()
	{
		return (m_TreeNode->mixedMask[0] || m_TreeNode->mixedMask[1] || m_TreeNode->mixedMask[2] || m_TreeNode->mixedMask[3]);
	}

	const bool* SerializedProperty::GetMixedMask()
	{
		return m_TreeNode->mixedMask;
	}

	void* SerializedProperty::GetHintData()
	{
		return m_TreeNode->fieldInfo->options.hintData;
	}

	const size_t SerializedProperty::GetListSize()
	{
		if (m_TreeNode->type == PropertyType::List)
		{
			return m_TreeNode->children.size();
		}
		return 0;
	}

	SerializedProperty SerializedProperty::GetListElement(const size_t& index)
	{
		return SerializedProperty(m_SerializedObject, m_TreeNode->children[index].get());
	}

	void SerializedProperty::InsertListElement(const size_t& index)
	{
		// TODO move into SerializedObject and use id instead of PropertyTreeNode*
		std::shared_ptr<PropertyTreeNode> childNode = m_SerializedObject->CreateChild(m_TreeNode);
		m_TreeNode->children.insert(m_TreeNode->children.begin() + index, childNode);
		for (size_t i = index; i < m_TreeNode->children.size(); ++i)
		{
			m_TreeNode->children[i]->index = i;
		}
		m_SerializedObject->AddModifiedProperty(m_TreeNode, PropertyModificationType::Insert, index);
	}

	void SerializedProperty::DeleteListElement(const size_t& index)
	{
		m_TreeNode->children[index]->isDeleted = true;
		m_SerializedObject->AddModifiedProperty(m_TreeNode, PropertyModificationType::Delete, index);
	}

	void SerializedProperty::MoveListElement(const size_t& fromIndex, const size_t& toIndex)
	{
		m_TreeNode->children.move_element(fromIndex, toIndex);
		for (size_t i = 0; i < m_TreeNode->children.size(); ++i)
		{
			m_TreeNode->children[i]->index = i;
		}
		m_SerializedObject->AddModifiedProperty(m_TreeNode, PropertyModificationType::Move, fromIndex, toIndex);
	}

	void SerializedProperty::ClearList()
	{
		for (auto& childNode : m_TreeNode->children)
		{
			childNode->isDeleted = true;
		}
		m_SerializedObject->AddModifiedProperty(m_TreeNode, PropertyModificationType::Clear);
	}

	const bool SerializedProperty::IsOverriden()
	{
		return m_TreeNode->isOverriden;
	}

	void SerializedProperty::ClearOverride()
	{
		m_SerializedObject->AddModifiedProperty(m_TreeNode, PropertyModificationType::ClearOverride);
	}

	const bool& SerializedProperty::GetBool()
	{
		return std::get<bool>(m_TreeNode->values[0]);
	}

	void SerializedProperty::SetBool(const bool& value)
	{
		for (size_t i = 0; i < m_TreeNode->values.size(); ++i)
		{
			m_TreeNode->values[i] = value;
		}
		m_SerializedObject->AddModifiedProperty(m_TreeNode);
	}

	const int& SerializedProperty::GetInt()
	{
		return std::get<int>(m_TreeNode->values[0]);
	}

	void SerializedProperty::SetInt(const int& value)
	{
		for (size_t i = 0; i < m_TreeNode->values.size(); ++i)
		{
			m_TreeNode->values[i] = value;
		}
		m_SerializedObject->AddModifiedProperty(m_TreeNode);
	}

	const float& SerializedProperty::GetFloat()
	{
		return std::get<float>(m_TreeNode->values[0]);
	}

	void SerializedProperty::SetFloat(const float& value)
	{
		for (size_t i = 0; i < m_TreeNode->values.size(); ++i)
		{
			m_TreeNode->values[i] = value;
		}
		m_SerializedObject->AddModifiedProperty(m_TreeNode);
	}

	const String& SerializedProperty::GetString()
	{
		return std::get<String>(m_TreeNode->values[0]);
	}

	void SerializedProperty::SetString(const String& value)
	{
		for (size_t i = 0; i < m_TreeNode->values.size(); ++i)
		{
			m_TreeNode->values[i] = value;
		}
		m_SerializedObject->AddModifiedProperty(m_TreeNode);
	}

	const Vector2& SerializedProperty::GetVector2()
	{
		return std::get<Vector2>(m_TreeNode->values[0]);
	}

	void SerializedProperty::SetVector2(const Vector2& value)
	{
		Vector2 targetValue = std::get<Vector2>(m_TreeNode->values[0]);
		bool* mask = m_TreeNode->mixedMask;
		m_TreeNode->values[0] = value;
		for (size_t i = 1; i < m_TreeNode->values.size(); ++i)
		{
			Vector2 nodeValue = std::get<Vector2>(m_TreeNode->values[i]);
			if (mask[0])
			{
				nodeValue.x = value.x;
			}
			if (mask[1])
			{
				nodeValue.y = value.y;
			}
			m_TreeNode->values[i] = nodeValue;
		}
		m_SerializedObject->AddModifiedProperty(m_TreeNode);
	}

	const Vector3& SerializedProperty::GetVector3()
	{
		return std::get<Vector3>(m_TreeNode->values[0]);
	}

	void SerializedProperty::SetVector3(const Vector3& value)
	{
		Vector3 targetValue = std::get<Vector3>(m_TreeNode->values[0]);
		bool* mask = m_TreeNode->mixedMask;
		m_TreeNode->values[0] = value;
		for (size_t i = 1; i < m_TreeNode->values.size(); ++i)
		{
			Vector3 nodeValue = std::get<Vector3>(m_TreeNode->values[i]);
			if (mask[0])
			{
				nodeValue.x = value.x;
			}
			if (mask[1])
			{
				nodeValue.y = value.y;
			}
			if (mask[2])
			{
				nodeValue.z = value.z;
			}
			m_TreeNode->values[i] = nodeValue;
		}
		m_SerializedObject->AddModifiedProperty(m_TreeNode);
	}

	const Vector4& SerializedProperty::GetVector4()
	{
		return std::get<Vector4>(m_TreeNode->values[0]);
	}

	void SerializedProperty::SetVector4(const Vector4& value)
	{
		Vector4 targetValue = std::get<Vector4>(m_TreeNode->values[0]);
		bool* mask = m_TreeNode->mixedMask;
		m_TreeNode->values[0] = value;
		for (size_t i = 1; i < m_TreeNode->values.size(); ++i)
		{
			Vector4 nodeValue = std::get<Vector4>(m_TreeNode->values[i]);
			if (mask[0])
			{
				nodeValue.x = value.x;
			}
			if (mask[1])
			{
				nodeValue.y = value.y;
			}
			if (mask[2])
			{
				nodeValue.z = value.z;
			}
			if (mask[3])
			{
				nodeValue.w = value.w;
			}
			m_TreeNode->values[i] = nodeValue;
		}
		m_SerializedObject->AddModifiedProperty(m_TreeNode);
	}

	const Quaternion& SerializedProperty::GetQuaternion()
	{
		return std::get<Quaternion>(m_TreeNode->values[0]);
	}

	void SerializedProperty::SetQuaternion(const Quaternion& value)
	{
		Quaternion targetValue = std::get<Quaternion>(m_TreeNode->values[0]);
		bool* mask = m_TreeNode->mixedMask;
		m_TreeNode->values[0] = value;
		for (size_t i = 1; i < m_TreeNode->values.size(); ++i)
		{
			Quaternion nodeValue = std::get<Quaternion>(m_TreeNode->values[i]);
			if (mask[0])
			{
				nodeValue.x = value.x;
			}
			if (mask[1])
			{
				nodeValue.y = value.y;
			}
			if (mask[2])
			{
				nodeValue.z = value.z;
			}
			if (mask[3])
			{
				nodeValue.w = value.w;
			}
			m_TreeNode->values[i] = nodeValue;
		}
		m_SerializedObject->AddModifiedProperty(m_TreeNode);
	}

	const Color& SerializedProperty::GetColor()
	{
		return std::get<Color>(m_TreeNode->values[0]);
	}

	void SerializedProperty::SetColor(const Color& value)
	{
		for (size_t i = 0; i < m_TreeNode->values.size(); ++i)
		{
			m_TreeNode->values[i] = value;
		}
		m_SerializedObject->AddModifiedProperty(m_TreeNode);
	}

	const ObjectPtr<Object>& SerializedProperty::GetObjectPtr()
	{
		return std::get<ObjectPtr<Object>>(m_TreeNode->values[0]);
	}

	void SerializedProperty::SetObjectPtr(const ObjectPtr<Object>& value)
	{
		for (size_t i = 0; i < m_TreeNode->values.size(); ++i)
		{
			m_TreeNode->values[i] = value;
		}
		m_SerializedObject->AddModifiedProperty(m_TreeNode);
	}

	const size_t& SerializedProperty::GetObjectPtrType()
	{
		return m_TreeNode->fieldInfo->options.objectType;
	}

	SerializedProperty::SerializedProperty(SerializedObject* serializedObject, PropertyTreeNode* treeNode)
	{
		m_SerializedObject = serializedObject;
		m_TreeNode = treeNode;
	}

	bool SerializedProperty::Next(const bool& enterChildren)
	{
		while (!m_Stack.empty())
		{
			std::pair<PropertyTreeNode*, uint32_t>& pair = m_Stack.top();
			if (pair.second < pair.first->children.size())
			{
				PropertyTreeNode* child = pair.first->children[pair.second++].get();
				if (child->isVisible && !child->isDeleted)
				{
					m_TreeNode = child;
					if (enterChildren)
					{
						m_Stack.push(std::make_pair(child, 0));
					}
					return true;
				}
			}
			else
			{
				m_Stack.pop();
				if (!m_Stack.empty())
				{
					return Next();
				}
				return false;
			}
		}
		return false;
	}

	SerializedProperty SerializedProperty::FindProperty(const String& name)
	{
		for (auto& child : m_TreeNode->children)
		{
			if (child->name == name)
			{
				return SerializedProperty(m_SerializedObject, child.get());
			}
		}
		return {};
	}
}
