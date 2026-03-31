#include "Blueberry\Serialization\BinaryReader.h"

#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Serialization\SerializationTree.h"

#include <fstream>

namespace Blueberry
{
	struct Context
	{
		uint32_t version;
		std::ifstream& stream;
		List<char>& keyBuffer;
	};

	void ReadNode(SerializationNodeRef ref, SerializationTree& tree, Context& context)
	{
		SerializationNode& node = ref.Get();
		uint32_t keyOffset;
		context.stream.read(reinterpret_cast<char*>(&keyOffset), sizeof(uint32_t));
		context.stream.read(reinterpret_cast<char*>(&node.flags), sizeof(uint8_t));
		node.key = String(context.keyBuffer.data() + keyOffset);
		if (node.flags & (SerializationFlags::MAP | SerializationFlags::SEQUENCE | SerializationFlags::FLOWMAP))
		{
			uint32_t childCount;
			context.stream.read(reinterpret_cast<char*>(&childCount), sizeof(uint32_t));
			for (uint32_t i = 0; i < childCount; ++i)
			{
				ReadNode(ref.AppendChild(), tree, context);
			}
		}
		else
		{
			uint32_t valueSize;
			context.stream.read(reinterpret_cast<char*>(&valueSize), sizeof(uint32_t));
			node.value.resize(valueSize);
			context.stream.read(node.value.data(), valueSize);
		}
	}

	void ReadObject(SerializationTree& tree, Context& context)
	{
		uint32_t rootNodeCount;
		if (context.version == 1)
		{
			uint32_t keyOffset;
			context.stream.read(reinterpret_cast<char*>(&keyOffset), sizeof(uint32_t));
			tree.typeName = String(context.keyBuffer.data() + keyOffset);
			tree.typeId = ClassDB::GetTypeId(tree.typeName);
		}
		else
		{
			size_t typeHash;
			context.stream.read(reinterpret_cast<char*>(&typeHash), sizeof(size_t));
			tree.typeId = ClassDB::GetInfo(typeHash)->id;
		}
		context.stream.read(reinterpret_cast<char*>(&tree.fileId), sizeof(FileId));
		context.stream.read(reinterpret_cast<char*>(&tree.isReference), sizeof(bool));
		context.stream.read(reinterpret_cast<char*>(&rootNodeCount), sizeof(uint32_t));
		SerializationNodeRef root = tree.GetRoot();
		for (uint32_t i = 0; i < rootNodeCount; ++i)
		{
			SerializationNodeRef child = root.AppendChild();
			ReadNode(child, tree, context);
		}
	}

	void BinaryReader::Read(List<SerializationTree>& trees, std::ifstream& stream)
	{
		List<char> keyBuffer;
		Context context = { 0, stream, keyBuffer };
		char header;
		uint32_t keyBufferSize, objectCount;
		stream.read(&header, sizeof(char));
		stream.read(reinterpret_cast<char*>(&context.version), sizeof(uint32_t));
		stream.read(reinterpret_cast<char*>(&keyBufferSize), sizeof(uint32_t));
		context.keyBuffer.resize(keyBufferSize);
		stream.read(context.keyBuffer.data(), keyBufferSize);
		stream.read(reinterpret_cast<char*>(&objectCount), sizeof(uint32_t));
		for (uint32_t i = 0; i < objectCount; ++i)
		{
			SerializationTree tree = {};
			tree.isText = false;
			ReadObject(tree, context);
			trees.push_back(tree);
		}
	}
}