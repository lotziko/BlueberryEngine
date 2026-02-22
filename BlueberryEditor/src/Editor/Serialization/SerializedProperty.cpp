#include "SerializedProperty.h"

#include "SerializedObject.h"
#include "Blueberry\Core\Object.h"
#include "Blueberry\Core\ClassDB.h"

namespace Blueberry
{
	size_t SerializedProperty::GetId()
	{
		return m_Id;
	}

	const String& SerializedProperty::GetName()
	{
		return Get()->name;
	}

	const BindingType SerializedProperty::GetType()
	{
		return Get()->bindingType;
	}

	const bool SerializedProperty::IsMixedValue()
	{
		PropertyTreeNode* node = Get();
		return (node->mixedMask[0] || node->mixedMask[1] || node->mixedMask[2] || node->mixedMask[3]);
	}

	const bool* SerializedProperty::GetMixedMask()
	{
		return Get()->mixedMask;
	}

	void* SerializedProperty::GetHintData()
	{
		return Get()->fieldInfo->options.hintData;
	}

	const size_t SerializedProperty::GetListSize()
	{
		PropertyTreeNode* node = Get();
		if (node->type == PropertyType::List)
		{
			return node->children.size();
		}
		return 0;
	}

	SerializedProperty SerializedProperty::GetListElement(const size_t& index)
	{
		return SerializedProperty(m_SerializedObject, m_SerializedObject->Get(m_Id)->children[index]);
	}

	void SerializedProperty::InsertListElement(const size_t& index)
	{
		// TODO move into SerializedObject and use id instead of PropertyTreeNode*
		PropertyTreeNode* node = Get();
		size_t child = m_SerializedObject->CreateChild(m_Id);
		node->children.insert(node->children.begin() + index, child);
		for (size_t i = index; i < node->children.size(); ++i)
		{
			m_SerializedObject->Get(node->children[i])->index = i;
		}
		m_SerializedObject->AddModifiedProperty(m_Id, PropertyModificationType::Insert, index);
	}

	void SerializedProperty::DeleteListElement(const size_t& index)
	{
		m_SerializedObject->DeleteListElement(m_Id, index);
	}

	void SerializedProperty::MoveListElement(const size_t& fromIndex, const size_t& toIndex)
	{
		m_SerializedObject->MoveListElement(m_Id, fromIndex, toIndex);
	}

	void SerializedProperty::ClearList()
	{
		m_SerializedObject->ClearList(m_Id);
	}

	const bool SerializedProperty::IsOverriden()
	{
		return m_SerializedObject->Get(m_Id)->isOverriden;
	}

	void SerializedProperty::ClearOverride()
	{
		m_SerializedObject->AddModifiedProperty(m_Id, PropertyModificationType::ClearOverride);
	}

	const bool& SerializedProperty::GetBool()
	{
		return std::get<bool>(Get()->values[0]);
	}

	void SerializedProperty::SetBool(const bool& value)
	{
		PropertyTreeNode* node = Get();
		for (size_t i = 0; i < node->values.size(); ++i)
		{
			node->values[i] = value;
		}
		m_SerializedObject->AddModifiedProperty(m_Id);
	}

	const int& SerializedProperty::GetInt()
	{
		return std::get<int>(Get()->values[0]);
	}

	void SerializedProperty::SetInt(const int& value)
	{
		PropertyTreeNode* node = Get();
		for (size_t i = 0; i < node->values.size(); ++i)
		{
			node->values[i] = value;
		}
		m_SerializedObject->AddModifiedProperty(m_Id);
	}

	const uint32_t& SerializedProperty::GetUint()
	{
		return std::get<uint32_t>(Get()->values[0]);
	}

	void SerializedProperty::SetUint(const uint32_t& value)
	{
		PropertyTreeNode* node = Get();
		for (size_t i = 0; i < node->values.size(); ++i)
		{
			node->values[i] = value;
		}
		m_SerializedObject->AddModifiedProperty(m_Id);
	}

	const float& SerializedProperty::GetFloat()
	{
		return std::get<float>(Get()->values[0]);
	}

	void SerializedProperty::SetFloat(const float& value)
	{
		PropertyTreeNode* node = Get();
		for (size_t i = 0; i < node->values.size(); ++i)
		{
			node->values[i] = value;
		}
		m_SerializedObject->AddModifiedProperty(m_Id);
	}

	const String& SerializedProperty::GetString()
	{
		return std::get<String>(Get()->values[0]);
	}

	void SerializedProperty::SetString(const String& value)
	{
		PropertyTreeNode* node = Get();
		for (size_t i = 0; i < node->values.size(); ++i)
		{
			node->values[i] = value;
		}
		m_SerializedObject->AddModifiedProperty(m_Id);
	}

	const Vector2& SerializedProperty::GetVector2()
	{
		return std::get<Vector2>(Get()->values[0]);
	}

	void SerializedProperty::SetVector2(const Vector2& value)
	{
		PropertyTreeNode* node = Get();
		Vector2 targetValue = std::get<Vector2>(node->values[0]);
		bool* mask = node->mixedMask;
		node->values[0] = value;
		for (size_t i = 1; i < node->values.size(); ++i)
		{
			Vector2 nodeValue = std::get<Vector2>(node->values[i]);
			if (mask[0])
			{
				nodeValue.x = value.x;
			}
			if (mask[1])
			{
				nodeValue.y = value.y;
			}
			node->values[i] = nodeValue;
		}
		m_SerializedObject->AddModifiedProperty(m_Id);
	}

	const Vector3& SerializedProperty::GetVector3()
	{
		return std::get<Vector3>(Get()->values[0]);
	}

	void SerializedProperty::SetVector3(const Vector3& value)
	{
		PropertyTreeNode* node = Get();
		Vector3 targetValue = std::get<Vector3>(node->values[0]);
		bool* mask = node->mixedMask;
		node->values[0] = value;
		for (size_t i = 1; i < node->values.size(); ++i)
		{
			Vector3 nodeValue = std::get<Vector3>(node->values[i]);
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
			node->values[i] = nodeValue;
		}
		m_SerializedObject->AddModifiedProperty(m_Id);
	}

	const Vector4& SerializedProperty::GetVector4()
	{
		return std::get<Vector4>(Get()->values[0]);
	}

	void SerializedProperty::SetVector4(const Vector4& value)
	{
		PropertyTreeNode* node = Get();
		Vector4 targetValue = std::get<Vector4>(node->values[0]);
		bool* mask = node->mixedMask;
		node->values[0] = value;
		for (size_t i = 1; i < node->values.size(); ++i)
		{
			Vector4 nodeValue = std::get<Vector4>(node->values[i]);
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
			node->values[i] = nodeValue;
		}
		m_SerializedObject->AddModifiedProperty(m_Id);
	}

	const Quaternion& SerializedProperty::GetQuaternion()
	{
		return std::get<Quaternion>(Get()->values[0]);
	}

	void SerializedProperty::SetQuaternion(const Quaternion& value)
	{
		PropertyTreeNode* node = Get();
		Quaternion targetValue = std::get<Quaternion>(node->values[0]);
		bool* mask = node->mixedMask;
		node->values[0] = value;
		for (size_t i = 1; i < node->values.size(); ++i)
		{
			Quaternion nodeValue = std::get<Quaternion>(node->values[i]);
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
			node->values[i] = nodeValue;
		}
		m_SerializedObject->AddModifiedProperty(m_Id);
	}

	const Color& SerializedProperty::GetColor()
	{
		return std::get<Color>(Get()->values[0]);
	}

	void SerializedProperty::SetColor(const Color& value)
	{
		PropertyTreeNode* node = Get();
		for (size_t i = 0; i < node->values.size(); ++i)
		{
			node->values[i] = value;
		}
		m_SerializedObject->AddModifiedProperty(m_Id);
	}

	const ObjectPtr<Object>& SerializedProperty::GetObjectPtr()
	{
		return std::get<ObjectPtr<Object>>(Get()->values[0]);
	}

	void SerializedProperty::SetObjectPtr(const ObjectPtr<Object>& value)
	{
		PropertyTreeNode* node = Get();
		for (size_t i = 0; i < node->values.size(); ++i)
		{
			node->values[i] = value;
		}
		m_SerializedObject->AddModifiedProperty(m_Id);
	}

	const size_t& SerializedProperty::GetObjectPtrType()
	{
		return Get()->fieldInfo->options.objectType;
	}

	SerializedProperty::SerializedProperty(SerializedObject* serializedObject, size_t id)
	{
		m_SerializedObject = serializedObject;
		m_Id = id;
	}

	PropertyTreeNode* SerializedProperty::Get()
	{
		return m_SerializedObject->Get(m_Id);
	}

	bool SerializedProperty::Next(const bool& enterChildren)
	{
		PropertyTreeNode* node = Get();
		while (!m_Stack.empty())
		{
			std::pair<size_t, uint32_t>& pair = m_Stack.top();
			PropertyTreeNode* stackNode = m_SerializedObject->Get(pair.first);
			if (pair.second < stackNode->children.size())
			{
				size_t child = stackNode->children[pair.second++];
				PropertyTreeNode* childNode = m_SerializedObject->Get(child);
				if (childNode->isVisible && !childNode->isDeleted)
				{
					m_Id = child;
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
		for (auto& child : Get()->children)
		{
			PropertyTreeNode* childNode = m_SerializedObject->Get(child);
			if (childNode->name == name)
			{
				return SerializedProperty(m_SerializedObject, child);
			}
		}
		return {};
	}

	SerializedObject* SerializedProperty::GetSerializedObject()
	{
		return m_SerializedObject;
	}
}
