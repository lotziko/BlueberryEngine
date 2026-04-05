#include "Blueberry\Serialization\SerializationTree.h"

#include "Blueberry\Tools\ByteConverter.h"

#include <charconv>

namespace Blueberry
{
	#define EMPTY_ID UINT64_MAX

	SerializationTreeFlags operator|(SerializationTreeFlags lhs, SerializationTreeFlags rhs)
	{
		return static_cast<SerializationTreeFlags>(static_cast<uint8_t>(lhs) | static_cast<uint8_t>(rhs));
	}

	SerializationTreeFlags& operator|=(SerializationTreeFlags& lhs, SerializationTreeFlags rhs)
	{
		return lhs = static_cast<SerializationTreeFlags>(static_cast<uint8_t>(lhs) | static_cast<uint8_t>(rhs));
	}

	SerializationTreeFlags operator&(SerializationTreeFlags lhs, SerializationTreeFlags rhs)
	{
		return static_cast<SerializationTreeFlags>(static_cast<uint8_t>(lhs) & static_cast<uint8_t>(rhs));
	}

	ChildIterator& ChildIterator::operator++()
	{
		childId = tree->GetNextSibling(childId);
		return *this;
	}

	ChildIterator& ChildIterator::operator--()
	{
		childId = tree->GetPreviousSibling(childId);
		return *this;
	}

	SerializationNodeRef ChildIterator::operator*() const
	{
		return { childId, tree };
	}

	SerializationNodeRef ChildIterator::operator->() const
	{
		return { childId, tree };
	}

	bool ChildIterator::operator!=(ChildIterator other) const
	{
		return childId != other.childId;
	}

	bool ChildIterator::operator==(ChildIterator other) const
	{
		return childId == other.childId;
	}

	ConstChildIterator& ConstChildIterator::operator++()
	{
		childId = tree->GetNextSibling(childId);
		return *this;
	}

	ConstChildIterator& ConstChildIterator::operator--()
	{
		childId = tree->GetPreviousSibling(childId);
		return *this;
	}

	SerializationNodeConstRef ConstChildIterator::operator*() const
	{
		return { childId, tree };
	}

	SerializationNodeConstRef ConstChildIterator::operator->() const
	{
		return { childId, tree };
	}

	bool ConstChildIterator::operator!=(ConstChildIterator other) const
	{
		return childId != other.childId;
	}

	bool ConstChildIterator::operator==(ConstChildIterator other) const
	{
		return childId == other.childId;
	}

	ChildIterator ChildView::begin() const
	{
		return ChildIterator(tree, childId);
	}

	ChildIterator ChildView::end() const
	{
		return ChildIterator(tree, UINT64_MAX);
	}

	ConstChildIterator ConstChildView::begin() const
	{
		return ConstChildIterator(tree, childId);
	}

	ConstChildIterator ConstChildView::end() const
	{
		return ConstChildIterator(tree, UINT64_MAX);
	}

	void ReadValue(SerializationNodeConstRef& ref, bool& value)
	{
		SerializationNode& node = ref.Get();
		if (ref.tree->isText)
		{
			const char* text = node.value.data();
			value = (strcmp(text, "1") == 0);
		}
		else
		{
			memcpy(&value, node.value.data(), sizeof(bool));
		}
	}

	void ReadValue(SerializationNodeConstRef& ref, float& value)
	{
		SerializationNode& node = ref.Get();
		if (ref.tree->isText)
		{
			const char* text = node.value.data();
			value = strtof(text, nullptr);
		}
		else
		{
			memcpy(&value, node.value.data(), sizeof(float));
		}
	}

	void ReadValue(SerializationNodeConstRef& ref, int& value)
	{
		SerializationNode& node = ref.Get();
		if (ref.tree->isText)
		{
			const char* text = node.value.data();
			value = static_cast<int>(strtol(text, nullptr, 10));
		}
		else
		{
			memcpy(&value, node.value.data(), sizeof(int));
		}
	}

	void ReadValue(SerializationNodeConstRef& ref, unsigned int& value)
	{
		SerializationNode& node = ref.Get();
		if (ref.tree->isText)
		{
			const char* text = node.value.data();
			value = static_cast<uint32_t>(strtol(text, nullptr, 10));
		}
		else
		{
			memcpy(&value, node.value.data(), sizeof(unsigned int));
		}
	}

	void ReadValue(SerializationNodeConstRef& ref, long& value)
	{
		SerializationNode& node = ref.Get();
		if (ref.tree->isText)
		{
			const char* text = node.value.data();
			value = strtol(text, nullptr, 10);
		}
		else
		{
			memcpy(&value, node.value.data(), sizeof(long));
		}
	}

	void ReadValue(SerializationNodeConstRef & ref, long long& value)
	{
		SerializationNode& node = ref.Get();
		if (ref.tree->isText)
		{
			const char* text = node.value.data();
			value = strtoll(text, nullptr, 10);
		}
		else
		{
			memcpy(&value, node.value.data(), sizeof(long long));
		}
	}

	void ReadValue(SerializationNodeConstRef& ref, unsigned long long& value)
	{
		SerializationNode& node = ref.Get();
		if (ref.tree->isText)
		{
			const char* text = node.value.data();
			value = strtoull(text, nullptr, 10);
		}
		else
		{
			memcpy(&value, node.value.data(), sizeof(unsigned long long));
		}
	}

	void ReadValue(SerializationNodeConstRef& ref, String& value)
	{
		SerializationNode& node = ref.Get();
		if (ref.tree->isText)
		{
			if (node.value.size() > 0)
			{
				value = String(node.value.data(), node.value.size() - 1);
			}
		}
		else
		{
			value = String(node.value.data(), node.value.size());
		}
	}

	void ReadValue(SerializationNodeConstRef& ref, Vector2& value)
	{
		ref["x"] >> value.x;
		ref["y"] >> value.y;
	}

	void ReadValue(SerializationNodeConstRef& ref, Vector2Int& value)
	{
		ref["x"] >> value.x;
		ref["y"] >> value.y;
	}

	void ReadValue(SerializationNodeConstRef& ref, Vector3& value)
	{
		ref["x"] >> value.x;
		ref["y"] >> value.y;
		ref["z"] >> value.z;
	}

	void ReadValue(SerializationNodeConstRef& ref, Vector3Int& value)
	{
		ref["x"] >> value.x;
		ref["y"] >> value.y;
		ref["z"] >> value.z;
	}

	void ReadValue(SerializationNodeConstRef& ref, Vector4& value)
	{
		ref["x"] >> value.x;
		ref["y"] >> value.y;
		ref["z"] >> value.z;
		ref["w"] >> value.w;
	}

	void ReadValue(SerializationNodeConstRef& ref, Vector4Int& value)
	{
		ref["x"] >> value.x;
		ref["y"] >> value.y;
		ref["z"] >> value.z;
		ref["w"] >> value.w;
	}

	void ReadValue(SerializationNodeConstRef& ref, Quaternion& value)
	{
		ref["x"] >> value.x;
		ref["y"] >> value.y;
		ref["z"] >> value.z;
		ref["w"] >> value.w;
	}

	void ReadValue(SerializationNodeConstRef& ref, Color& value)
	{
		bool isOld = ref["x"].IsValid();
		if (isOld)
		{
			ref["x"] >> value.x;
			ref["y"] >> value.y;
			ref["z"] >> value.z;
			ref["w"] >> value.w;
		}
		else
		{
			ref["r"] >> value.x;
			ref["g"] >> value.y;
			ref["b"] >> value.z;
			ref["a"] >> value.w;
		}
	}

	void ReadValue(SerializationNodeConstRef& ref, AABB& value)
	{
		Vector3 center, extents;
		ref["center"] >> center;
		ref["extents"] >> extents;
		value.Center = center;
		value.Extents = extents;
	}

	void ReadValue(SerializationNodeConstRef& ref, Matrix& value)
	{
		SerializationNode& node = ref.Get();
		if (ref.tree->isText)
		{
			size_t size = node.value.size() - 1;
			ByteConverter::HexStringToBytes(node.value.data(), value.m, size);
		}
		else
		{
			memcpy(&value, node.value.data(), sizeof(Matrix));
		}
	}

	void ReadValue(SerializationNodeConstRef& ref, Guid& value)
	{
		SerializationNode& node = ref.Get();
		if (ref.tree->isText)
		{
			size_t size = node.value.size() - 1;
			ByteConverter::HexStringToBytes(node.value.data(), value.data, size);
		}
		else
		{
			memcpy(&value, node.value.data(), sizeof(Guid));
		}
	}

	void ReadValue(SerializationNodeConstRef& ref, ObjectPtrData& value)
	{
		ref["fileId"] >> value.fileId;
		SerializationNodeConstRef guidRef = ref["guid"];
		if (guidRef.IsValid())
		{
			guidRef >> value.guid;
		}
		else
		{
			value.guid = {};
		}
	}

	void ReadValue(SerializationNodeConstRef& ref, DataWrapper<ByteData>& value)
	{
		SerializationNode& node = ref.Get();
		if (ref.tree->isText)
		{
			if (node.value.size() > 0)
			{
				size_t size = node.value.size() - 1;
				value.reference.resize(size / (2 * sizeof(uint8_t)));
				ByteConverter::HexStringToBytes(node.value.data(), value.reference.data(), size);
			}
		}
		else
		{
			value.reference.resize(node.value.size());
			memcpy(value.reference.data(), node.value.data(), node.value.size());
		}
	}

	void ReadValue(SerializationNodeConstRef& ref, DataWrapper<List<int>>& value)
	{
		SerializationNode& node = ref.Get();
		if (ref.tree->isText)
		{
			if (node.value.size() > 0)
			{
				size_t size = node.value.size() - 1;
				value.reference.resize(size / (2 * sizeof(int)));
				ByteConverter::HexStringToBytes(node.value.data(), value.reference.data(), size);
			}
		}
		else
		{
			value.reference.resize(node.value.size());
			memcpy(value.reference.data(), node.value.data(), node.value.size());
		}
	}

	void ReadValue(SerializationNodeConstRef& ref, DataWrapper<List<float>>& value)
	{
		SerializationNode& node = ref.Get();
		if (ref.tree->isText)
		{
			if (node.value.size() > 0)
			{
				size_t size = node.value.size() - 1;
				value.reference.resize(size / (2 * sizeof(float)));
				ByteConverter::HexStringToBytes(node.value.data(), value.reference.data(), size);
			}
		}
		else
		{
			value.reference.resize(node.value.size());
			memcpy(value.reference.data(), node.value.data(), node.value.size());
		}
	}

	void WriteValue(SerializationNodeRef& ref, bool& value)
	{
		SerializationNode& node = ref.Get();
		if (ref.tree->isText)
		{
			node.value.resize(2);
			node.value[0] = value ? '1' : '0';
			node.value[1] = '\0';
		}
		else
		{
			node.value.resize(sizeof(bool));
			memcpy(node.value.data(), &value, sizeof(bool));
		}
	}

	void WriteValue(SerializationNodeRef& ref, float& value)
	{
		SerializationNode& node = ref.Get();
		if (ref.tree->isText)
		{
			char buffer[64];
			auto result = std::to_chars(buffer, buffer + 64, value, std::chars_format::general);
			size_t size = result.ptr - buffer;
			node.value.resize(size + 1, '\0');
			memcpy(node.value.data(), buffer, size);
		}
		else
		{
			node.value.resize(sizeof(float));
			memcpy(node.value.data(), &value, sizeof(float));
		}
	}

	void WriteValue(SerializationNodeRef& ref, int& value)
	{
		SerializationNode& node = ref.Get();
		if (ref.tree->isText)
		{
			char buffer[64];
			auto result = std::to_chars(buffer, buffer + 64, value);
			size_t size = result.ptr - buffer;
			node.value.resize(size + 1, '\0');
			memcpy(node.value.data(), buffer, size);
		}
		else
		{
			node.value.resize(sizeof(int));
			memcpy(node.value.data(), &value, sizeof(int));
		}
	}

	void WriteValue(SerializationNodeRef& ref, unsigned int& value)
	{
		SerializationNode& node = ref.Get();
		if (ref.tree->isText)
		{
			char buffer[64];
			auto result = std::to_chars(buffer, buffer + 64, value);
			size_t size = result.ptr - buffer;
			node.value.resize(size + 1, '\0');
			memcpy(node.value.data(), buffer, size);
		}
		else
		{
			node.value.resize(sizeof(unsigned int));
			memcpy(node.value.data(), &value, sizeof(unsigned int));
		}
	}

	void WriteValue(SerializationNodeRef& ref, long& value)
	{
		SerializationNode& node = ref.Get();
		if (ref.tree->isText)
		{
			char buffer[64];
			auto result = std::to_chars(buffer, buffer + 64, value);
			size_t size = result.ptr - buffer;
			node.value.resize(size + 1, '\0');
			memcpy(node.value.data(), buffer, size);
		}
		else
		{
			node.value.resize(sizeof(long));
			memcpy(node.value.data(), &value, sizeof(long));
		}
	}

	void WriteValue(SerializationNodeRef& ref, long long& value)
	{
		SerializationNode& node = ref.Get();
		if (ref.tree->isText)
		{
			char buffer[64];
			auto result = std::to_chars(buffer, buffer + 64, value);
			size_t size = result.ptr - buffer;
			node.value.resize(size + 1, '\0');
			memcpy(node.value.data(), buffer, size);
		}
		else
		{
			node.value.resize(sizeof(long long));
			memcpy(node.value.data(), &value, sizeof(long long));
		}
	}

	void WriteValue(SerializationNodeRef& ref, unsigned long long& value)
	{
		SerializationNode& node = ref.Get();
		if (ref.tree->isText)
		{
			char buffer[64];
			auto result = std::to_chars(buffer, buffer + 64, value);
			size_t size = result.ptr - buffer;
			node.value.resize(size + 1, '\0');
			memcpy(node.value.data(), buffer, size);
		}
		else
		{
			node.value.resize(sizeof(unsigned long long));
			memcpy(node.value.data(), &value, sizeof(unsigned long long));
		}
	}

	void WriteValue(SerializationNodeRef& ref, String& value)
	{
		SerializationNode& node = ref.Get();
		if (value.size() > 0)
		{
			if (ref.tree->isText)
			{
				node.value.resize(value.size() + 1, '\0');
				memcpy(node.value.data(), value.data(), value.size());
			}
			else
			{
				node.value.resize(value.size());
				memcpy(node.value.data(), value.data(), value.size());
			}
		}
	}

	void WriteValue(SerializationNodeRef& ref, Vector2& value)
	{
		ref |= SerializationTreeFlags::FLOWMAP;
		ref["x"] << value.x;
		ref["y"] << value.y;
	}

	void WriteValue(SerializationNodeRef& ref, Vector2Int& value)
	{
		ref |= SerializationTreeFlags::FLOWMAP;
		ref["x"] << value.x;
		ref["y"] << value.y;
	}

	void WriteValue(SerializationNodeRef& ref, Vector3& value)
	{
		ref |= SerializationTreeFlags::FLOWMAP;
		ref["x"] << value.x;
		ref["y"] << value.y;
		ref["z"] << value.z;
	}

	void WriteValue(SerializationNodeRef& ref, Vector3Int& value)
	{
		ref |= SerializationTreeFlags::FLOWMAP;
		ref["x"] << value.x;
		ref["y"] << value.y;
		ref["z"] << value.z;
	}

	void WriteValue(SerializationNodeRef& ref, Vector4& value)
	{
		ref |= SerializationTreeFlags::FLOWMAP;
		ref["x"] << value.x;
		ref["y"] << value.y;
		ref["z"] << value.z;
		ref["w"] << value.w;
	}

	void WriteValue(SerializationNodeRef& ref, Vector4Int& value)
	{
		ref |= SerializationTreeFlags::FLOWMAP;
		ref["x"] << value.x;
		ref["y"] << value.y;
		ref["z"] << value.z;
		ref["w"] << value.w;
	}

	void WriteValue(SerializationNodeRef& ref, Quaternion& value)
	{
		ref |= SerializationTreeFlags::FLOWMAP;
		ref["x"] << value.x;
		ref["y"] << value.y;
		ref["z"] << value.z;
		ref["w"] << value.w;
	}

	void WriteValue(SerializationNodeRef& ref, Color& value)
	{
		ref |= SerializationTreeFlags::FLOWMAP;
		ref["r"] << value.x;
		ref["g"] << value.y;
		ref["b"] << value.z;
		ref["a"] << value.w;
	}

	void WriteValue(SerializationNodeRef& ref, AABB& value)
	{
		ref |= SerializationTreeFlags::FLOWMAP;
		ref["center"] << Vector3(value.Center);
		ref["extents"] << Vector3(value.Extents);
	}

	void WriteValue(SerializationNodeRef& ref, Matrix& value)
	{
		SerializationNode& node = ref.Get();
		if (ref.tree->isText)
		{
			node.value.resize(sizeof(Matrix) * 2 + 1, '\0');
			ByteConverter::BytesToHexString(value.m, node.value.data(), sizeof(Matrix));
		}
		else
		{
			node.value.resize(sizeof(Matrix));
			memcpy(node.value.data(), &value, sizeof(Matrix));
		}
	}

	void WriteValue(SerializationNodeRef& ref, Guid& value)
	{
		SerializationNode& node = ref.Get();
		if (ref.tree->isText)
		{
			node.value.resize(16 * 2 + 1, '\0');
			ByteConverter::BytesToHexString(value.data, node.value.data(), sizeof(value.data));
		}
		else
		{
			node.value.resize(sizeof(Guid));
			memcpy(node.value.data(), &value, sizeof(Guid));
		}
	}

	void WriteValue(SerializationNodeRef& ref, ObjectPtrData& value)
	{
		ref |= SerializationTreeFlags::FLOWMAP;
		ref["fileId"] << value.fileId;
		if (value.guid.data[0] > 0)
		{
			ref["guid"] << value.guid;
		}
	}

	void WriteValue(SerializationNodeRef& ref, DataWrapper<ByteData>& value)
	{
		SerializationNode& node = ref.Get();
		if (ref.tree->isText)
		{
			node.value.resize(value.reference.size() * 2 * sizeof(uint8_t) + 1, '\0');
			ByteConverter::BytesToHexString(value.reference.data(), node.value.data(), value.reference.size() * sizeof(uint8_t));
		}
		else
		{
			node.value.resize(value.reference.size() * sizeof(uint8_t));
			memcpy(node.value.data(), value.reference.data(), value.reference.size() * sizeof(uint8_t));
		}
	}

	void WriteValue(SerializationNodeRef& ref, DataWrapper<List<int>>& value)
	{
		SerializationNode& node = ref.Get();
		if (ref.tree->isText)
		{
			node.value.resize(value.reference.size() * 2 * sizeof(int) + 1, '\0');
			ByteConverter::BytesToHexString(value.reference.data(), node.value.data(), value.reference.size() * sizeof(int));
		}
		else
		{
			node.value.resize(value.reference.size() * sizeof(int));
			memcpy(node.value.data(), value.reference.data(), value.reference.size() * sizeof(int));
		}
	}

	void WriteValue(SerializationNodeRef& ref, DataWrapper<List<float>>& value)
	{
		SerializationNode& node = ref.Get();
		if (ref.tree->isText)
		{
			node.value.resize(value.reference.size() * 2 * sizeof(float) + 1, '\0');
			ByteConverter::BytesToHexString(value.reference.data(), node.value.data(), value.reference.size() * sizeof(float));
		}
		else
		{
			node.value.resize(value.reference.size() * sizeof(float));
			memcpy(node.value.data(), value.reference.data(), value.reference.size() * sizeof(float));
		}
	}

	void SerializationNodeRef::operator|=(SerializationTreeFlags flags)
	{
		tree->nodes[id].flags |= flags;
	}

	SerializationNodeRef SerializationNodeRef::operator[](const char* name)
	{
		size_t childId = tree->AppendChild(id);
		tree->nodes[childId].key = name;
		return { childId, tree };
	}

	SerializationNode& SerializationNodeRef::Get()
	{
		return tree->nodes[id];
	}

	SerializationNodeRef SerializationNodeRef::GetNextSibling()
	{
		return { tree->nodes[id].nextSibling, tree };
	}

	SerializationNodeRef SerializationNodeRef::AppendChild()
	{
		return { tree->AppendChild(id), tree };
	}

	ChildView SerializationNodeRef::GetChildren()
	{
		return ChildView(tree, tree->nodes[id].firstChild);
	}

	SerializationTree::SerializationTree()
	{
		size_t rootId = Allocate();
		nodes[rootId].key = "Root";
	}

	SerializationNodeRef SerializationTree::operator[](const char* name)
	{
		for (size_t i = 0; i != EMPTY_ID; GetNextSibling(i))
		{
			if (nodes[i].key == name)
			{
				return { i, this };
			}
		}
		return { EMPTY_ID, this };
	}

	SerializationNodeRef SerializationTree::GetRoot()
	{
		return { 0, this };
	}

	SerializationNodeConstRef SerializationTree::GetConstRoot()
	{
		return { 0, this };
	}

	size_t SerializationTree::FindChild(size_t id, const char* name)
	{
		SerializationNode& node = nodes[id];
		if (node.firstChild == EMPTY_ID)
		{
			return EMPTY_ID;
		}
		for (size_t i = node.firstChild; i != EMPTY_ID; i = GetNextSibling(i))
		{
			if (nodes[i].key == name)
			{
				return i;
			}
		}
		return EMPTY_ID;
	}

	size_t SerializationTree::FindChild(size_t id, size_t index)
	{
		SerializationNode& node = nodes[id];
		if (node.firstChild == EMPTY_ID)
		{
			return EMPTY_ID;
		}
		size_t count = 0;
		for (size_t i = node.firstChild; i != EMPTY_ID; i = GetNextSibling(i))
		{
			if (count++ == index)
			{
				return i;
			}
		}
		return EMPTY_ID;
	}

	size_t SerializationTree::GetFirstChild(size_t id)
	{
		return nodes[id].firstChild;
	}

	size_t SerializationTree::GetPreviousSibling(size_t id)
	{
		return nodes[id].previousSibling;
	}

	size_t SerializationTree::GetNextSibling(size_t id)
	{
		return nodes[id].nextSibling;
	}

	size_t SerializationTree::Allocate()
	{
		size_t index = nodes.size();
		nodes.emplace_back();
		return index;
	}

	size_t SerializationTree::InsertChild(size_t parentId, size_t afterId)
	{
		size_t childId = Allocate();
		SetHierarchy(childId, parentId, afterId);
		return childId;
	}

	size_t SerializationTree::AppendChild(size_t parentId)
	{
		return InsertChild(parentId, nodes[parentId].lastChild);
	}

	void SerializationTree::SetHierarchy(size_t childId, size_t parentId, size_t previousSiblingId)
	{
		SerializationNode& child = nodes[childId];
		child.parent = parentId;
		child.previousSibling = EMPTY_ID;
		child.nextSibling = EMPTY_ID;

		if (parentId == EMPTY_ID)
		{
			return;
		}

		size_t nextSiblingId = previousSiblingId != EMPTY_ID ? GetNextSibling(previousSiblingId) : GetFirstChild(parentId);
		
		if (previousSiblingId != EMPTY_ID)
		{
			SerializationNode& previousSibling = nodes[previousSiblingId];
			child.previousSibling = previousSiblingId;
			previousSibling.nextSibling = childId;
		}

		if (nextSiblingId != EMPTY_ID)
		{
			SerializationNode& nextSibling = nodes[nextSiblingId];
			child.nextSibling = nextSiblingId;
			nextSibling.previousSibling = childId;
		}

		SerializationNode& parent = nodes[parentId];

		if (parent.firstChild == EMPTY_ID)
		{
			parent.firstChild = childId;
			parent.lastChild = childId;
		}
		else
		{
			if (child.nextSibling == parent.firstChild)
			{
				parent.firstChild = childId;
			}

			if (child.previousSibling == parent.lastChild)
			{
				parent.lastChild = childId;
			}
		}
	}

	SerializationNodeConstRef SerializationNodeConstRef::operator[](const char* name)
	{
		return { tree->FindChild(id, name), tree };
	}

	SerializationNodeConstRef SerializationNodeConstRef::operator[](const size_t index)
	{
		return { tree->FindChild(id, index), tree };
	}

	SerializationNode& SerializationNodeConstRef::Get()
	{
		return tree->nodes[id];
	}

	SerializationNodeConstRef SerializationNodeConstRef::GetNextSibling()
	{
		return SerializationNodeConstRef();
	}

	ConstChildView SerializationNodeConstRef::GetChildren()
	{
		return ConstChildView(tree, tree->nodes[id].firstChild);
	}

	bool SerializationNodeConstRef::IsValid()
	{
		return id != EMPTY_ID;
	}
}
