#include "SerializedProperty.h"

#include "SerializedObject.h"
#include "Blueberry\Core\Object.h"
#include "Blueberry\Core\ClassDB.h"

namespace Blueberry
{
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

	const uint32_t SerializedProperty::GetArraySize()
	{
		if (m_TreeNode->type == PropertyType::List)
		{
			return static_cast<uint32_t>(m_TreeNode->children.size());
		}
		return 0;
	}

	SerializedProperty SerializedProperty::GetArrayElement(const uint32_t& index)
	{
		return SerializedProperty(m_SerializedObject, m_TreeNode->children[index].get());
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
		m_SerializedObject->AddModifiedProperty(this);
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
		m_SerializedObject->AddModifiedProperty(this);
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
		m_SerializedObject->AddModifiedProperty(this);
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
		m_SerializedObject->AddModifiedProperty(this);
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
		m_SerializedObject->AddModifiedProperty(this);
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
		m_SerializedObject->AddModifiedProperty(this);
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
		m_SerializedObject->AddModifiedProperty(this);
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
		m_SerializedObject->AddModifiedProperty(this);
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
		m_SerializedObject->AddModifiedProperty(this);
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
		m_SerializedObject->AddModifiedProperty(this);
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

	bool SerializedProperty::Next()
	{
		while (!m_Stack.empty())
		{
			std::pair<PropertyTreeNode*, uint32_t>& pair = m_Stack.top();
			if (pair.second < pair.first->children.size())
			{
				PropertyTreeNode* child = pair.first->children[pair.second++].get();
				if (child->isVisible)
				{
					m_TreeNode = child;
					m_Stack.push(std::make_pair(child, 0));
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
