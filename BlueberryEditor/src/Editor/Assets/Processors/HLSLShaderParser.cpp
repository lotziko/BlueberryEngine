#include "bbpch.h"
#include "HLSLShaderParser.h"

#include "Blueberry\Tools\FileHelper.h"

namespace Blueberry
{
	bool HLSLShaderParser::Parse(const std::string& path, std::string& shaderCode, RawShaderOptions& options)
	{
		std::string shaderData;
		FileHelper::Load(shaderData, path);

		if (shaderData.find("Shader") == 0)
		{
			auto shaderBeginTag = "HLSLBEGIN";
			auto shaderEndTag = "HLSLEND";
			size_t shaderBegin;
			size_t shaderEnd;

			if ((shaderBegin = shaderData.find(shaderBeginTag)) == std::string::npos || (shaderEnd = shaderData.find(shaderEndTag, shaderBegin)) == std::string::npos)
			{
				return false;
			}
			shaderBegin += sizeof(shaderBeginTag) + 1;
			shaderCode = shaderData.substr(shaderBegin, shaderEnd - shaderBegin);

			size_t optionsTagBegin;
			if ((optionsTagBegin = shaderData.find("Options")) != std::string::npos)
			{
				auto optionsBeginTag = "{";
				auto optionsEndTag = "}";

				size_t optionsBegin;
				size_t optionsEnd;
				if ((optionsBegin = shaderData.find(optionsBeginTag, optionsTagBegin)) != std::string::npos && (optionsEnd = shaderData.find(optionsEndTag, optionsBegin)) != std::string::npos)
				{
					optionsBegin += 3;
					optionsEnd -= 2;

					for (size_t i = optionsBegin; i < optionsEnd; ++i)
					{
						if (shaderData[i] < 32)
						{
							continue;
						}
						else
						{
							std::string optionKey;
							for (size_t j = i; j < optionsEnd; ++j)
							{
								if (shaderData[j] == ' ')
								{
									optionKey = shaderData.substr(i, j - i);
									i = j + 1;
									break;
								}
							}
							std::string optionValue;
							for (size_t j = i; j < optionsEnd; ++j)
							{
								if (shaderData[j] < 32)
								{
									optionValue = shaderData.substr(i, j - i);
									i = j + 1;
									break;
								}
							}
							options.insert_or_assign(optionKey, optionValue);
						}
					}
				}
			}
		}
		else
		{
			shaderCode = shaderData;
		}
		return true;
	}
}
