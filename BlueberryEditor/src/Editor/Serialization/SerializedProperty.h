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

		size_t GetId();
		const String& GetName();
		const BindingType GetType();
		const bool IsMixedValue();
		const bool* GetMixedMask();
		void* GetHintData();

		const size_t GetListSize();
		SerializedProperty GetListElement(const size_t& index);
		void InsertListElement(const size_t& index);
		void DeleteListElement(const size_t& index);
		void MoveListElement(const size_t& fromIndex, const size_t& toIndex);
		void ClearList();

		const bool IsOverriden();
		void ClearOverride();

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

		bool Next(const bool& enterChildren = true);
		SerializedProperty FindProperty(const String& name);

		SerializedObject* GetSerializedObject();

	private:
		SerializedProperty(SerializedObject* serializedObject, size_t id);
		PropertyTreeNode* Get();

	private:
		SerializedObject* m_SerializedObject;
		size_t m_Id;
		std::stack<std::pair<size_t, uint32_t>> m_Stack;

		friend class SerializedObject;
	};

	template<typename T>
	inline T SerializedProperty::GetEnum()
	{
		return static_cast<T>(std::get<int>(Get()->values[0]));
	}
}