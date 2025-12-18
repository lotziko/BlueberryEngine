#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\ObjectPtr.h"
#include "Blueberry\Core\Variant.h"

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

	enum class PropertyModificationType
	{
		Value,
		Insert,
		Delete,
		Move,
		Clear,
		ClearOverride
	};

	using PropertyValue = Variant;

	struct PropertyTreeNode
	{
		String name;
		size_t index;
		bool isVisible;
		bool isDeleted;
		bool isOverriden;
		BindingType bindingType;
		const FieldInfo* fieldInfo;
		PropertyType type;
		bool mixedMask[4];
		List<PropertyValue> values;

		std::weak_ptr<PropertyTreeNode> parent;
		List<std::shared_ptr<PropertyTreeNode>> children;
	};

	struct PropertyModification
	{
		PropertyTreeNode* node;
		PropertyModificationType type;
		size_t index1;
		size_t index2;
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

		void Update();

		bool ApplyModifiedProperties();

	private:
		void BuildTree();
		void BuildProperties(std::weak_ptr<PropertyTreeNode> parent, const ClassInfo* classInfo, const List<void*>& targets);
		void BuildList(std::weak_ptr<PropertyTreeNode> parent, const FieldInfo& fieldInfo, const List<void*>& targets);
		void ReadTree();
		void ReadProperties(std::weak_ptr<PropertyTreeNode> parent, const ClassInfo* classInfo, const List<void*>& targets);
		void ReadList(std::weak_ptr<PropertyTreeNode> parent, const FieldInfo& fieldInfo, const List<void*>& targets);
		void AddModifiedProperty(PropertyTreeNode* node, const PropertyModificationType& type = PropertyModificationType::Value, const size_t& index1 = 0, const size_t& index2 = 0);
		
		std::shared_ptr<PropertyTreeNode> CreateChild(PropertyTreeNode* parent);

		void FindPath(PropertyTreeNode* node, List<void*>& result, size_t& offset);
		void ApplyModification(const PropertyModification& modification);

	private:
		const ClassInfo* m_ClassInfo = nullptr;
		List<Object*> m_Targets = {};
		List<PropertyModification> m_Modifications = {};
		std::shared_ptr<PropertyTreeNode> m_Root;
		List<bool> m_IsPrefabInstance = {};

		friend class SerializedProperty;
	};
}