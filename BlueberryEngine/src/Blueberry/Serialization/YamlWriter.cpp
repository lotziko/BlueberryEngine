#include "Blueberry\Serialization\YamlWriter.h"

#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Serialization\SerializationTree.h"

#include <fstream>

namespace Blueberry
{
	void WriteHeader(std::ofstream& stream, SerializationTree& tree)
	{
		stream << "--- &" << tree.fileId;
		if (tree.isReference)
		{
			stream << " reference";
		}
		stream << "\n" << ClassDB::GetInfo(tree.typeId)->name << ":\n";
	}

	void Indent(std::ofstream& stream, uint32_t value)
	{
		stream << String(value, ' ');
	}

	void WriteValue(std::ofstream& stream, SerializationNode& node)
	{
		if (node.value.size() > 0)
		{
			stream << node.value.data();
		}
		else
		{
			stream << "''";
		}
	}

	void WriteFlowmap(std::ofstream& stream, SerializationNode& node, SerializationTree& tree)
	{
		stream << "{";
		for (size_t i = node.firstChild; i != UINT64_MAX;)
		{
			SerializationNode& elementNode = tree.nodes[i];
			stream << elementNode.key << ": ";
			if ((elementNode.flags & SerializationTreeFlags::FLOWMAP) != SerializationTreeFlags::NONE)
			{
				WriteFlowmap(stream, elementNode, tree);
			}
			else
			{
				WriteValue(stream, elementNode);
			}
			if (elementNode.nextSibling != UINT64_MAX)
			{
				stream << ", ";
			}
			i = elementNode.nextSibling;
		}
		stream << "}";
	}

	void WriteNode(std::ofstream& stream, SerializationNode& node, SerializationTree& tree, uint32_t indent, bool& isSequenceElement)
	{
		if (indent > 0)
		{
			if (isSequenceElement)
			{
				if ((node.flags & SerializationTreeFlags::MAP) == SerializationTreeFlags::NONE)
				{
					Indent(stream, indent - 2);
					stream << "- ";
					isSequenceElement = false;
				}
			}
			else
			{
				Indent(stream, indent);
			}
		}
		if (node.key.size() > 0)
		{
			stream << node.key << ": ";
		}
		if ((node.flags & SerializationTreeFlags::FLOWMAP) != SerializationTreeFlags::NONE)
		{
			WriteFlowmap(stream, node, tree);
			stream << "\n";
		}
		else if ((node.flags & SerializationTreeFlags::MAP) != SerializationTreeFlags::NONE)
		{
			if (node.key.size() > 0)
			{
				stream << "\n";
			}
			if (!isSequenceElement)
			{
				indent += 2;
			}
			for (size_t i = node.firstChild; i != UINT64_MAX;)
			{
				SerializationNode& mapElementNode = tree.nodes[i];
				WriteNode(stream, mapElementNode, tree, indent, isSequenceElement);
				i = mapElementNode.nextSibling;
			}
		}
		else if ((node.flags & SerializationTreeFlags::SEQUENCE) != SerializationTreeFlags::NONE)
		{
			if (node.firstChild != UINT64_MAX)
			{
				stream << "\n";
				for (size_t i = node.firstChild; i != UINT64_MAX;)
				{
					SerializationNode& sequenceElementNode = tree.nodes[i];
					isSequenceElement = true;
					WriteNode(stream, sequenceElementNode, tree, indent + 2, isSequenceElement);
					i = sequenceElementNode.nextSibling;
				}
			}
			else
			{
				stream << "[]\n";
			}
		}
		else if (node.flags == SerializationTreeFlags::NONE)
		{
			WriteValue(stream, node);
			stream << "\n";
		}
	}

	void WriteObject(std::ofstream& stream, SerializationTree& tree, bool hasHeaders)
	{
		uint32_t indent = hasHeaders ? 2 : 0;
		bool isSequenceElement = false;
		for (size_t i = tree.nodes[0].firstChild; i != UINT64_MAX;)
		{
			SerializationNode& node = tree.nodes[i];
			WriteNode(stream, node, tree, indent, isSequenceElement);
			i = node.nextSibling;
		}
	}

	void YamlWriter::Write(List<SerializationTree>& trees, std::ofstream& stream, bool hasHeaders)
	{
		if (hasHeaders)
		{
			stream << "%YAML\n";
		}
		for (SerializationTree& tree : trees)
		{
			if (hasHeaders)
			{
				WriteHeader(stream, tree);
			}
			WriteObject(stream, tree, hasHeaders);
		}
	}
}