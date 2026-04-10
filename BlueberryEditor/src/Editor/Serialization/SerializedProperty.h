#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\Variant.h"
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
		const String& GetDisplayName();
		BindingType GetType();
		bool IsMixedValue();
		bool* GetMixedMask();
		void* GetHintData();

		size_t GetListSize();
		SerializedProperty GetListElement(size_t index);
		void InsertListElement(size_t index);
		void DeleteListElement(size_t index);
		void MoveListElement(size_t fromIndex, size_t toIndex);
		void ClearList();

		bool IsOverriden();
		void ClearOverride();

		bool GetBool();
		void SetBool(bool value);

		int GetInt();
		void SetInt(int value);

		uint32_t GetUint();
		void SetUint(uint32_t value);

		float GetFloat();
		void SetFloat(float value);

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

		TypeId GetObjectPtrType();

		size_t GetDepth();

		bool Next(bool enterChildren = true);
		SerializedProperty FindProperty(const String& name);

		SerializedObject* GetSerializedObject();

	private:
		SerializedProperty(SerializedObject* serializedObject, size_t id);
		PropertyTreeNode* Get();

	private:
		SerializedObject* m_SerializedObject;
		size_t m_Id = 0;
		size_t m_Depth = 0;

		friend class SerializedObject;
	};

	template<typename T>
	inline T SerializedProperty::GetEnum()
	{
		return static_cast<T>(std::get<int>(Get()->values[0]));
	}
}