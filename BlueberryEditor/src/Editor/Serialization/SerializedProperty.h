#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\ObjectPtr.h"

namespace Blueberry
{
	enum class BindingType;
	class SerializedObject;
	struct PropertyTreeNode;

	class SerializedProperty
	{
	public:
		BB_OVERRIDE_NEW_DELETE
		
		SerializedProperty() = default;

		const String& GetName();
		const BindingType GetType();
		const bool IsMixedValue();
		const bool* GetMixedMask();
		void* GetHintData();

		const uint32_t GetArraySize();
		SerializedProperty GetArrayElement(const uint32_t& index);

		const bool& GetBool();
		void SetBool(const bool& value);

		const int& GetInt();
		void SetInt(const int& value);

		const float& GetFloat();
		void SetFloat(const float& value);

		template<typename T>
		T GetEnum();

		const String& GetString();
		void SetString(const String& value);

		const Vector2& GetVector2();
		void SetVector2(const Vector2& value);

		const Vector3& GetVector3();
		void SetVector3(const Vector3& value);

		const Vector4& GetVector4();
		void SetVector4(const Vector4& value);

		const Quaternion& GetQuaternion();
		void SetQuaternion(const Quaternion& value);

		const Color& GetColor();
		void SetColor(const Color& value);

		const ObjectPtr<Object>& GetObjectPtr();
		void SetObjectPtr(const ObjectPtr<Object>& value);

		const size_t& GetObjectPtrType();

		bool Next();
		SerializedProperty FindProperty(const String& name);

	private:
		SerializedProperty(SerializedObject* serializedObject, PropertyTreeNode* treeNode);

	private:
		SerializedObject* m_SerializedObject;
		PropertyTreeNode* m_TreeNode;
		std::stack<std::pair<PropertyTreeNode*, uint32_t>> m_Stack;

		friend class SerializedObject;
	};

	template<typename T>
	inline T SerializedProperty::GetEnum()
	{
		return static_cast<T>(std::get<int>(m_TreeNode->values[0]));
	}
}