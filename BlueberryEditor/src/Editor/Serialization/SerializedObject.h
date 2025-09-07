#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\ObjectPtr.h"

#include <variant>

namespace Blueberry
{
	class Object;
	class SerializedProperty;
	struct ClassInfo;
	struct FieldInfo;
	enum class BindingType;

	enum class PropertyType
	{
		Root,
		Value,
		List,
		Data
	};

	using PropertyValue = std::variant<bool, int, float, String, Vector2, Vector3, Vector4, Quaternion, Color, ObjectPtr<Object>>;

	struct PropertyTreeNode
	{
		String name;
		uint32_t index;
		bool isVisible;
		BindingType bindingType;
		const FieldInfo* fieldInfo;
		PropertyType type;
		bool mixedMask[4];
		List<PropertyValue> values;

		std::weak_ptr<PropertyTreeNode> parent;
		List<std::shared_ptr<PropertyTreeNode>> children;
	};

	class SerializedObject : public std::enable_shared_from_this<SerializedObject>
	{
	public:
		BB_OVERRIDE_NEW_DELETE

		SerializedObject() = default;
		SerializedObject(Object* target);
		SerializedObject(List<Object*> targets);

		SerializedProperty FindProperty(const String& name);
		SerializedProperty GetIterator();

		Object* GetTarget();
		const List<Object*>& GetTargets();

		bool ApplyModifiedProperties();

	private:
		void BuildTree();
		void BuildProperties(std::weak_ptr<PropertyTreeNode> parent, const ClassInfo* classInfo, const List<void*>& targets);
		void BuildList(std::weak_ptr<PropertyTreeNode> parent, const FieldInfo& fieldInfo, const List<void*>& targets);
		void AddModifiedProperty(SerializedProperty* property);
		void ApplyValues(PropertyTreeNode* node);

	private:
		const ClassInfo* m_ClassInfo = nullptr;
		List<Object*> m_Targets = {};
		List<PropertyTreeNode*> m_ModifiedNodes = {};
		std::shared_ptr<PropertyTreeNode> m_Root;

		friend class SerializedProperty;
	};
}