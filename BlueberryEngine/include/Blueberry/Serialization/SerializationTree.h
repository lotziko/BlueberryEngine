#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\Object.h"
#include "Blueberry\Core\Guid.h"
#include "Blueberry\Core\Structs.h"

namespace Blueberry
{
	enum SerializationFlags : uint8_t
	{
		NONE = 0,
		SEQUENCE = 1,
		MAP = 2,
		VALUE = 4,
		FLOWMAP = 8,
	};

	struct SerializationNodeRef;
	struct SerializationNodeConstRef;
	struct SerializationNode;
	struct SerializationTree;

	struct ChildIterator
	{
		ChildIterator(SerializationTree* tree, size_t childId) : tree(tree), childId(childId) 
		{
		}

		ChildIterator& operator++();
		ChildIterator& operator--();

		SerializationNodeRef operator*() const;
		SerializationNodeRef operator->() const;

		bool operator!= (ChildIterator other) const;
		bool operator== (ChildIterator other) const;

		SerializationTree* tree;
		size_t childId;
	};

	struct ConstChildIterator
	{
		ConstChildIterator(SerializationTree* tree, size_t childId) : tree(tree), childId(childId)
		{
		}

		ConstChildIterator& operator++();
		ConstChildIterator& operator--();

		SerializationNodeConstRef operator*() const;
		SerializationNodeConstRef operator->() const;

		bool operator!= (ConstChildIterator other) const;
		bool operator== (ConstChildIterator other) const;

		SerializationTree* tree;
		size_t childId;
	};

	struct ChildView
	{
		ChildView(SerializationTree* tree, size_t childId) : tree(tree), childId(childId) {}

		ChildIterator begin() const;
		ChildIterator end() const;

		SerializationTree* tree;
		size_t childId;
	};

	struct ConstChildView
	{
		ConstChildView(SerializationTree* tree, size_t childId) : tree(tree), childId(childId) {}

		ConstChildIterator begin() const;
		ConstChildIterator end() const;

		SerializationTree* tree;
		size_t childId;
	};

	struct SerializationNodeRef
	{
		template<class Type>
		void operator<< (Type& value);

		void operator|= (SerializationFlags flags);

		SerializationNodeRef operator[] (const char* name);

		SerializationNode& Get();
		SerializationNodeRef GetNextSibling();
		SerializationNodeRef AppendChild();
		ChildView GetChildren();

		size_t id;
		SerializationTree* tree;
	};

	struct SerializationNodeConstRef
	{
		template<class Type>
		void operator>> (Type& value);

		SerializationNodeConstRef operator[] (const char* name);
		SerializationNodeConstRef operator[] (const size_t index);

		SerializationNode& Get();
		SerializationNodeConstRef GetNextSibling();
		ConstChildView GetChildren();

		bool IsValid();

		size_t id;
		SerializationTree* tree;
	};

	struct SerializationNode
	{
		String key;
		List<char> value;

		size_t parent = UINT64_MAX;
		size_t firstChild = UINT64_MAX;
		size_t lastChild = UINT64_MAX;
		size_t nextSibling = UINT64_MAX;
		size_t previousSibling = UINT64_MAX;

		uint8_t flags = 0;
	};

	struct SerializationTree
	{
		SerializationTree();

		SerializationNodeRef operator[] (const char* name);
		SerializationNodeRef GetRoot();
		SerializationNodeConstRef GetConstRoot();

		size_t FindChild(size_t id, const char* name);
		size_t FindChild(size_t id, size_t index);
		size_t GetFirstChild(size_t id);
		size_t GetPreviousSibling(size_t id);
		size_t GetNextSibling(size_t id);

		size_t Allocate();
		size_t InsertChild(size_t parentId, size_t afterId);
		size_t AppendChild(size_t parentId);

		void SetHierarchy(size_t childId, size_t parentId, size_t previousSiblingId);

		TypeId typeId;
		String typeName;
		FileId fileId;
		ObjectId objectId;
		List<SerializationNode> nodes;
		bool isReference = false;
		bool isText;
	};

	void ReadValue(SerializationNodeConstRef& ref, bool& value);
	void ReadValue(SerializationNodeConstRef& ref, float& value);
	void ReadValue(SerializationNodeConstRef& ref, int& value);
	void ReadValue(SerializationNodeConstRef& ref, unsigned int& value);
	void ReadValue(SerializationNodeConstRef& ref, long& value);
	void ReadValue(SerializationNodeConstRef& ref, long long& value);
	void ReadValue(SerializationNodeConstRef& ref, unsigned long long& value);
	void ReadValue(SerializationNodeConstRef& ref, String& value);
	void ReadValue(SerializationNodeConstRef& ref, Vector2& value);
	void ReadValue(SerializationNodeConstRef& ref, Vector2Int& value);
	void ReadValue(SerializationNodeConstRef& ref, Vector3& value);
	void ReadValue(SerializationNodeConstRef& ref, Vector3Int& value);
	void ReadValue(SerializationNodeConstRef& ref, Vector4& value);
	void ReadValue(SerializationNodeConstRef& ref, Vector4Int& value);
	void ReadValue(SerializationNodeConstRef& ref, Quaternion& value);
	void ReadValue(SerializationNodeConstRef& ref, Color& value);
	void ReadValue(SerializationNodeConstRef& ref, AABB& value);
	void ReadValue(SerializationNodeConstRef& ref, Matrix& value);
	void ReadValue(SerializationNodeConstRef& ref, Guid& value);
	void ReadValue(SerializationNodeConstRef& ref, ObjectPtrData& value);
	void ReadValue(SerializationNodeConstRef& ref, DataWrapper<ByteData>& value);
	void ReadValue(SerializationNodeConstRef& ref, DataWrapper<List<int>>& value);
	void ReadValue(SerializationNodeConstRef& ref, DataWrapper<List<float>>& value);

	void WriteValue(SerializationNodeRef& ref, bool& value);
	void WriteValue(SerializationNodeRef& ref, float& value);
	void WriteValue(SerializationNodeRef& ref, int& value);
	void WriteValue(SerializationNodeRef& ref, unsigned int& value);
	void WriteValue(SerializationNodeRef& ref, long& value);
	void WriteValue(SerializationNodeRef& ref, long long& value);
	void WriteValue(SerializationNodeRef& ref, unsigned long long& value);
	void WriteValue(SerializationNodeRef& ref, String& value);
	void WriteValue(SerializationNodeRef& ref, Vector2& value);
	void WriteValue(SerializationNodeRef& ref, Vector2Int& value);
	void WriteValue(SerializationNodeRef& ref, Vector3& value);
	void WriteValue(SerializationNodeRef& ref, Vector3Int& value);
	void WriteValue(SerializationNodeRef& ref, Vector4& value);
	void WriteValue(SerializationNodeRef& ref, Vector4Int& value);
	void WriteValue(SerializationNodeRef& ref, Quaternion& value);
	void WriteValue(SerializationNodeRef& ref, Color& value);
	void WriteValue(SerializationNodeRef& ref, AABB& value);
	void WriteValue(SerializationNodeRef& ref, Matrix& value);
	void WriteValue(SerializationNodeRef& ref, Guid& value);
	void WriteValue(SerializationNodeRef& ref, ObjectPtrData& value);
	void WriteValue(SerializationNodeRef& ref, DataWrapper<ByteData>& value);
	void WriteValue(SerializationNodeRef& ref, DataWrapper<List<int>>& value);
	void WriteValue(SerializationNodeRef& ref, DataWrapper<List<float>>& value);

	template<class Type>
	inline void SerializationNodeRef::operator<<(Type& value)
	{
		WriteValue(*this, value);
	}

	template<class Type>
	inline void SerializationNodeConstRef::operator>>(Type& value)
	{
		ReadValue(*this, value);
	}
}