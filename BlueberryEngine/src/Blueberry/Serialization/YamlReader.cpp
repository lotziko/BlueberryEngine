#include "Blueberry\Serialization\YamlReader.h"

#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Serialization\SerializationTree.h"

#include <fstream>
#include <cctype>

namespace Blueberry
{
	void SkipWhitespace(const char*& p)
	{
		while (*p && std::isspace((unsigned char)*p))
		{
			++p;
		}
	}

	uint32_t GetIndent(const String& line)
	{
		uint32_t i = 0;
		while (i < line.size() && line[i] == ' ')
		{
			++i;
		}
		return i;
	}

	bool IsSequenceItem(const String& line, int indent)
	{
		return indent < line.size() && line[indent] == '-';
	}

	int GetRealIndent(int indent, bool isSequence)
	{
		return isSequence ? indent + 2 : indent;
	}

	struct ParentData
	{
		size_t id;
		size_t indent;
	};

	bool ReadHeader(std::ifstream& stream, SerializationTree& tree)
	{
		// Fileid
		String line;
		std::getline(stream, line);

		const char* p = line.c_str();
		SkipWhitespace(p);

		// Document start
		if (!(p[0] == '-' && p[1] == '-' && p[2] == '-'))
		{
			return false;
		}
		p += 3;
		SkipWhitespace(p);

		// Anchor
		if (*p != '&')
		{
			return false;
		}
		++p;

		FileId fileId = 0;
		const char* start = p;
		while (*p >= '0' && *p <= '9')
		{
			fileId = fileId * 10 + (*p - '0');
			++p;
		}

		if (p == start)
		{
			return false;
		}

		tree.fileId = fileId;
		tree.isReference = line.find("reference") != std::string::npos;

		// Object type
		std::getline(stream, line);
		p = line.c_str();
		SkipWhitespace(p);

		start = p;
		if (!std::isalpha((unsigned char)*p))
		{
			return false;
		}

		while (std::isalnum((unsigned char)*p) || *p == '_')
		{
			++p;
		}

		tree.typeName = String(start, size_t(p - start));
		tree.typeId = ClassDB::GetTypeId(tree.typeName);

		SkipWhitespace(p);

		if (*p != ':')
		{
			return false;
		}

		++p;

		SkipWhitespace(p);
		if (*p != '\n' && *p != '\r' && *p != '\0')
		{
			return false;
		}

		return true;
	}

	bool ReadObject(std::ifstream& stream, SerializationTree& tree)
	{
		tree.isText = true;
		size_t rootId = 0;
		std::stack<ParentData> parentsStack;
		parentsStack.push({ rootId, 0 });
		while (!stream.eof())
		{
			// Document end
			if (stream.peek() == '-')
			{
				break;
			}

			String line;
			std::getline(stream, line);
			if (line.length() == 0)
			{
				continue;
			}

			size_t dashPos = std::string::npos;
			size_t colonPos = std::string::npos;
			size_t firstFlowPos = std::string::npos;
			size_t lastFlowPos = std::string::npos;

			for (size_t i = 0; i < line.length(); i++)
			{
				switch (line[i])
				{
				case '-':
				{
					if (dashPos == std::string::npos)
					{
						bool noChars = true;
						for (size_t j = i - 1; j > 0; --j)
						{
							if (line[j] != ' ')
							{
								noChars = false;
								break;
							}
						}
						if (noChars)
						{
							dashPos = i;
						}
					}
				}
				break;
				case ':':
				{
					if (colonPos == std::string::npos)
					{
						colonPos = i;
					}
				}
				break;
				case '{':
				{
					if (firstFlowPos == std::string::npos)
					{
						firstFlowPos = i;
					}
				}
				break;
				case '}':
				{
					lastFlowPos = i;
				}
				break;
				}
			}

			size_t firstCharPos = line.find_first_not_of(" -\t\f\v\n\r");
			size_t lastCharPos = line.find_last_not_of(" \t\f\v\n\r");

			if (firstCharPos == std::string::npos)
			{
				continue;
			}

			while (parentsStack.size() != 0 && parentsStack.top().indent >= firstCharPos)
			{
				parentsStack.pop();
			}

			if (dashPos != std::string::npos) // Sequence element
			{
				while (parentsStack.size() != 0 && parentsStack.top().indent >= dashPos + 1)
				{
					parentsStack.pop();
				}

				if (colonPos != std::string::npos)
				{
					size_t elementId = tree.AppendChild(parentsStack.top().id);
					SerializationNode& elementNode = tree.nodes[elementId];
					parentsStack.push({ elementId, dashPos + 1 });
				}
				else
				{
					size_t valueId = tree.AppendChild(parentsStack.size() > 0 ? parentsStack.top().id : rootId);
					SerializationNode& valueNode = tree.nodes[valueId];
					valueNode.flags = SerializationTreeFlags::VALUE;

					String value = line.substr(firstCharPos, lastCharPos - firstCharPos + 1);
					value.erase(0, value.find_first_not_of(" \t\f\v\n\r"));
					valueNode.value.resize(value.size() + 1, '\0');
					memcpy(valueNode.value.data(), value.data(), value.size());
					continue;
				}
			}

			String key = line.substr(firstCharPos, ((colonPos <= lastCharPos) ? colonPos : lastCharPos + 1) - firstCharPos);

			if (colonPos == lastCharPos && colonPos != std::string::npos) // Start of map or sequence
			{
				size_t mapStartId = tree.AppendChild(parentsStack.size() > 0 ? parentsStack.top().id : rootId);
				tree.nodes[mapStartId].key = key;
				parentsStack.push({ mapStartId, firstCharPos });
				continue;
			}

			if (lastCharPos != std::string::npos) // Value
			{
				if (firstFlowPos != std::string::npos && lastFlowPos != std::string::npos && lastFlowPos - firstFlowPos > 1)
				{
					String flowLastKey = firstCharPos < firstFlowPos ? key : "";
					String flowLineValue = line.substr(firstFlowPos, firstFlowPos - lastFlowPos);
					size_t offset = 0;

					for (size_t i = 0; i < flowLineValue.length(); ++i)
					{
						bool isKeyStart = false;
						switch (flowLineValue[i])
						{
						case '{':
						{
							size_t flowId;
							if (dashPos != std::string::npos && colonPos > firstFlowPos && i == 0) // Sequence already created a node
							{
								flowId = parentsStack.top().id;
							}
							else
							{
								flowId = tree.AppendChild(parentsStack.size() > 0 ? parentsStack.top().id : rootId);
								parentsStack.push({ flowId, dashPos != std::string::npos ? (dashPos + 1) : firstCharPos });
							}
							SerializationNode& flowNode = tree.nodes[flowId];
							flowNode.key = flowLastKey;
							flowNode.flags = SerializationTreeFlags::FLOWMAP;
							isKeyStart = true;
						}
						break;
						case '}':
						{
							parentsStack.pop();
							isKeyStart = true;
						}
						break;
						case ',':
						{
							isKeyStart = true;
						}
						break;
						case ':':
						{
							flowLastKey = flowLineValue.substr(offset, i - offset);
							size_t valueStartPos = flowLineValue.find_first_not_of(" :", i);
							size_t valueEndPos = std::string::npos;
							for (size_t j = valueStartPos; j < flowLineValue.length(); ++j)
							{
								char flowChar = flowLineValue[j];
								if (flowChar == '{')
								{
									valueEndPos = std::string::npos;
									break;
								}
								else if (flowChar == ',' || flowChar == ' ' || flowChar == '}')
								{
									valueEndPos = j;
									break;
								}
								else if (flowChar == '\'' || flowChar == '[')
								{
									valueEndPos = valueStartPos;
									break;
								}
							}
							if (valueEndPos != std::string::npos)
							{
								size_t flowChildId = tree.AppendChild(parentsStack.size() > 0 ? parentsStack.top().id : rootId);
								SerializationNode& flowChildNode = tree.nodes[flowChildId];
								flowChildNode.key = flowLastKey;
								flowChildNode.flags = SerializationTreeFlags::VALUE;
								String flowValue = valueEndPos - valueStartPos > 0 ? flowLineValue.substr(valueStartPos, valueEndPos - valueStartPos) : "";
								flowChildNode.value.resize(flowValue.size() + 1, '\0');
								memcpy(flowChildNode.value.data(), flowValue.data(), flowValue.size());
							}
						}
						break;
						}
						if (isKeyStart)
						{
							offset = flowLineValue.find_first_not_of(" {},", i);
						}
					}
				}
				else if (colonPos < lastCharPos)
				{
					size_t valueId = tree.AppendChild(parentsStack.size() > 0 ? parentsStack.top().id : rootId);
					SerializationNode& valueNode = tree.nodes[valueId];
					valueNode.key = key;
					valueNode.flags = SerializationTreeFlags::VALUE;

					String value = line.substr(colonPos + 1, lastCharPos - colonPos);
					value.erase(0, value.find_first_not_of(" \t\f\v\n\r"));
					valueNode.value.resize(value.size() + 1, '\0');
					memcpy(valueNode.value.data(), value.data(), value.size());
				}
			}
		}
		return true;
	}

	void YamlReader::Read(List<SerializationTree>& trees, std::ifstream& stream, bool hasHeaders)
	{
		SerializationTree tree = {};
		if (hasHeaders)
		{
			while (!stream.eof())
			{
				if (ReadHeader(stream, tree) && ReadObject(stream, tree))
				{
					trees.push_back(std::move(tree));
					tree = {};
				}
			}
		}
		else
		{
			if (ReadObject(stream, tree))
			{
				trees.push_back(std::move(tree));
			}
		}
	}
}
