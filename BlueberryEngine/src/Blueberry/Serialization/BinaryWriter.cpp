#include "Blueberry\Serialization\BinaryWriter.h"

#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Serialization\SerializationTree.h"

#include <fstream>

namespace Blueberry
{
	struct Context
	{
		Dictionary<std::string_view, uint32_t>& keyToOffset;
		StringStream& keyStream;
		StringStream& dataStream;
		size_t streamSize = 0;
	};

	void WriteNode(SerializationNode& node, SerializationTree& tree, Context& context)
	{
		uint32_t keyOffset = 0;
		auto it = context.keyToOffset.find(node.key.c_str());
		if (it != context.keyToOffset.end())
		{
			keyOffset = it->second;
		}
		else
		{
			keyOffset = static_cast<uint32_t>(context.streamSize);
			context.keyToOffset[node.key] = keyOffset;
			context.keyStream << node.key << '\0';
			context.streamSize += node.key.size() + 1;
		}
		context.dataStream.write(reinterpret_cast<char*>(&keyOffset), sizeof(uint32_t));
		context.dataStream.write(reinterpret_cast<char*>(&node.flags), sizeof(uint8_t));
		if ((node.flags & (SerializationTreeFlags::MAP | SerializationTreeFlags::SEQUENCE | SerializationTreeFlags::FLOWMAP)) != SerializationTreeFlags::NONE)
		{
			uint32_t childCount = 0;
			for (size_t i = node.firstChild; i != UINT64_MAX; i = tree.GetNextSibling(i))
			{
				childCount++;
			}
			context.dataStream.write(reinterpret_cast<char*>(&childCount), sizeof(uint32_t));
			if (childCount > 0)
			{
				for (size_t i = node.firstChild; i != UINT64_MAX;)
				{
					SerializationNode& elementNode = tree.nodes[i];
					WriteNode(elementNode, tree, context);
					i = elementNode.nextSibling;
				}
			}
		}
		else
		{
			uint32_t valueSize = static_cast<uint32_t>(node.value.size());
			context.dataStream.write(reinterpret_cast<char*>(&valueSize), sizeof(valueSize));
			context.dataStream.write(node.value.data(), node.value.size());
		}
	}

	void WriteObject(SerializationTree& tree, Context& context, bool hasGuids)
	{
		uint32_t rootNodeCount = 0;
		for (size_t i = tree.nodes[0].firstChild; i != UINT64_MAX; i = tree.GetNextSibling(i))
		{
			rootNodeCount++;
		}
		const String& typeName = ClassDB::GetInfo(tree.typeId)->name;
		uint32_t keyOffset = 0;
		auto it = context.keyToOffset.find(typeName.c_str());
		if (it != context.keyToOffset.end())
		{
			keyOffset = it->second;
		}
		else
		{
			keyOffset = static_cast<uint32_t>(context.streamSize);
			context.keyToOffset[typeName] = keyOffset;
			context.keyStream << typeName << '\0';
			context.streamSize += typeName.size() + 1;
		}
		context.dataStream.write(reinterpret_cast<char*>(&keyOffset), sizeof(uint32_t));
		context.dataStream.write(reinterpret_cast<char*>(&tree.fileId), sizeof(FileId));
		if (hasGuids)
		{
			context.dataStream.write(reinterpret_cast<char*>(&tree.guid), sizeof(Guid));
		}
		context.dataStream.write(reinterpret_cast<char*>(&tree.isReference), sizeof(bool));
		context.dataStream.write(reinterpret_cast<char*>(&rootNodeCount), sizeof(uint32_t));
		for (size_t i = tree.nodes[0].firstChild; i != UINT64_MAX;)
		{
			SerializationNode& node = tree.nodes[i];
			WriteNode(node, tree, context);
			i = node.nextSibling;
		}
	}

	void BinaryWriter::Write(List<SerializationTree>& trees, std::ofstream& stream, bool hasGuids)
	{
		uint32_t version = 1;
		stream << 'B';
		stream.write(reinterpret_cast<char*>(&version), sizeof(uint32_t));
		Dictionary<std::string_view, uint32_t> keyToOffset;
		StringStream keyStream;
		StringStream dataStream;
		Context context = { keyToOffset, keyStream, dataStream, 0 };
		for (SerializationTree& tree : trees)
		{
			WriteObject(tree, context, hasGuids);
		}
		uint32_t keyBufferSize = static_cast<uint32_t>(context.streamSize);
		stream.write(reinterpret_cast<char*>(&keyBufferSize), sizeof(uint32_t));
		stream << keyStream.rdbuf();
		uint32_t objectCount = static_cast<uint32_t>(trees.size());
		stream.write(reinterpret_cast<char*>(&objectCount), sizeof(uint32_t));
		stream << dataStream.rdbuf();
	}
}