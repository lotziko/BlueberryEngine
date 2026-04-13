#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\ObjectPtr.h"
#include "Blueberry\Core\Variant.h"
#include "Blueberry\Events\Event.h"

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
		String displayName;
		size_t index;
		bool isVisible;
		bool isDeleted;
		bool isOverriden;
		BindingType bindingType;
		const FieldInfo* fieldInfo;
		PropertyType type;
		bool mixedMask[4];
		List<PropertyValue> values;

		size_t parent;
		List<size_t> children;
	};

	struct PropertyModification
	{
		size_t id;
		PropertyModificationType type;
		size_t index1;
		size_t index2;
	};

	class ObjectUpdateEventArgs
	{
	public:
		ObjectUpdateEventArgs(Object* object) : m_Object(object)
		{
		}

		Object* GetObject() const;

	private:
		Object* m_Object;
	};

	using ObjectUpdateEvent = Event<const ObjectUpdateEventArgs>;

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

		bool IsValid();
		void Update();

		bool ApplyModifiedProperties();

		static ObjectUpdateEvent& GetObjectUpdated();

	private:
		void BuildTree();
		void BuildProperties(size_t parent, const ClassInfo* classInfo, const List<void*>& targets, const bool& read);
		void BuildList(size_t parent, const FieldInfo& fieldInfo, const List<void*>& targets, const bool& read);
		void ReadTree();
		void ReadProperties(size_t parent, const ClassInfo* classInfo, const List<void*>& targets);
		void ReadList(size_t parent, const FieldInfo& fieldInfo, const List<void*>& targets);
		void AddModifiedProperty(size_t id, const PropertyModificationType& type = PropertyModificationType::Value, size_t index1 = 0, size_t index2 = 0);
		
		void InsertListElement(size_t id, size_t index);
		void DeleteListElement(size_t id, size_t index);
		void MoveListElement(size_t id, size_t fromIndex, size_t toIndex);
		void ClearList(size_t id);

		size_t Allocate();
		size_t CreateChild(size_t parent);
		PropertyTreeNode* Get(size_t id);

		String GetNodePath(size_t id);
		void FindPath(PropertyTreeNode* node, List<void*>& result);
		void ApplyModification(const PropertyModification& modification);

	private:
		const ClassInfo* m_ClassInfo = nullptr;
		List<Object*> m_Targets = {};
		List<ObjectPtr<Object>> m_TargetPtrs = {};
		List<PropertyModification> m_Modifications = {};
		List<PropertyTreeNode> m_Nodes = {};
		List<bool> m_IsPrefabInstance = {};

		static ObjectUpdateEvent s_ObjectUpdated;

		friend class SerializedProperty;
	};
}