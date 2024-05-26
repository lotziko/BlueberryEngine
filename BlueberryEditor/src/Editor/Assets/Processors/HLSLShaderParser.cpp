#include "bbpch.h"
#include "HLSLShaderParser.h"

#include "Blueberry\Tools\FileHelper.h"

namespace Blueberry
{
	bool CheckBrackets(const std::string& text)
	{
		int counter = 0;
		for (size_t i = 0; i < text.length(); ++i)
		{
			if (text[i] == '{')
			{
				++counter;
			}
			else if (text[i] == '}')
			{
				--counter;
			}
		}
		return counter == 0;
	}

	bool ParseBlock(const std::string& text, const std::string& name, const std::string& begin, const std::string& end, std::string& result)
	{
		// Check brackets
		if (!CheckBrackets(text))
		{
			return false;
		}

		// Check name
		size_t offset = 0;
		if (name.length() > 0)
		{
			if ((offset = text.find(name, 0)) != std::string::npos)
			{
				offset += name.length();
			}
			else
			{
				return false;
			}
		}

		// Check begin
		if ((offset = text.find(begin, offset)) != std::string::npos)
		{
			offset += begin.length();
		}
		else
		{
			return false;
		}

		//Check end
		uint8_t counter = 1;
		size_t endOffset = offset;
		while (endOffset < text.length())
		{
			size_t nextBegin = text.find(begin, endOffset);
			size_t nextEnd = text.find(end, endOffset);
			
			if (nextBegin < nextEnd)
			{
				++counter;
				endOffset = nextBegin + begin.length();
			}
			else
			{
				--counter;
				endOffset = nextEnd + end.length();
			}
			if (counter == 0)
			{
				endOffset = nextEnd;
				break;
			}
		}

		if (counter == 0)
		{
			result = text.substr(offset, endOffset - offset);
			return true;
		}
		return false;
	}

	CullMode ParseCullMode(const std::string& name)
	{
		if (name == "Front")
		{
			return CullMode::Front;
		}
		else if (name == "Back")
		{
			return CullMode::Back;
		}
		return CullMode::None;
	}

	BlendMode ParseBlendMode(const std::string& name)
	{
		if (name == "Zero")
		{
			return BlendMode::Zero;
		}
		else if (name == "SrcAlpha")
		{
			return BlendMode::SrcAlpha;
		}
		else if (name == "OneMinusSrcAlpha")
		{
			return BlendMode::OneMinusSrcAlpha;
		}
		return BlendMode::One;
	}

	ZWrite ParseZWrite(const std::string& name)
	{
		if (name == "Off")
		{
			return ZWrite::Off;
		}
		return ZWrite::On;
	}

	void ParseShaderData(const std::map<std::string, std::string>& rawOptions, const std::map<std::string, std::pair<std::string, std::string>>& rawProperties, ShaderData& data)
	{
		auto& cullModeIt = rawOptions.find("Cull");
		if (cullModeIt != rawOptions.end())
		{
			data.SetCullMode(ParseCullMode(cullModeIt->second));
		}

		auto& blendSrcIt = rawOptions.find("BlendSrc");
		if (blendSrcIt != rawOptions.end())
		{
			data.SetBlendSrc(ParseBlendMode(blendSrcIt->second));
		}

		auto& blendDstIt = rawOptions.find("BlendDst");
		if (blendSrcIt != rawOptions.end())
		{
			data.SetBlendDst(ParseBlendMode(blendDstIt->second));
		}

		auto& zWriteIt = rawOptions.find("ZWrite");
		if (zWriteIt != rawOptions.end())
		{
			data.SetZWrite(ParseZWrite(zWriteIt->second));
		}

		std::vector<DataPtr<TextureParameterData>> textureParameters;
		for (auto& rawProperty : rawProperties)
		{
			if (rawProperty.second.first == "Texture2D")
			{
				TextureParameterData* parameter = new TextureParameterData();
				parameter->SetName(rawProperty.first);
				// TODO index
				textureParameters.emplace_back(DataPtr<TextureParameterData>(parameter));
			}
		}
		data.SetTextureParameters(textureParameters);
	}

	bool HLSLShaderParser::Parse(const std::string& path, std::string& shaderCode, ShaderData& data)
	{
		std::string shaderData;
		FileHelper::Load(shaderData, path);

		std::string shaderBlock;
		if (ParseBlock(shaderData, "Shader", "{", "}", shaderBlock))
		{
			std::string codeBlock;
			if (ParseBlock(shaderBlock, "", "HLSLBEGIN", "HLSLEND", codeBlock))
			{
				shaderCode = codeBlock;
			}
			else
			{
				return false;
			}

			std::map<std::string, std::string> options;
			std::map<std::string, std::pair<std::string, std::string>> properties;

			std::string optionsBlock;
			if (ParseBlock(shaderBlock, "Options", "{", "}", optionsBlock))
			{
				// Based on https://www.geeksforgeeks.org/regex_iterator-function-in-c-stl/
				std::regex optionRegex("([\\w-]+)\\s*([\\w-]+)[\r?\n]");
				auto optionsStart = std::sregex_iterator(optionsBlock.begin(), optionsBlock.end(), optionRegex);
				auto optionsEnd = std::sregex_iterator();

				for (std::regex_iterator i = optionsStart; i != optionsEnd; ++i)
				{
					std::smatch match = *i;
					if (match.size() == 3)
					{
						options.insert_or_assign(match[1], match[2]);
					}
				}
			}

			std::string propertiesBlock;
			if (ParseBlock(shaderBlock, "Properties", "{", "}", propertiesBlock))
			{
				std::regex optionRegex("([\\w-]+)\\s*([\\w-]+)\\s*[\\=]\\s*(.*)");
				auto optionsStart = std::sregex_iterator(propertiesBlock.begin(), propertiesBlock.end(), optionRegex);
				auto optionsEnd = std::sregex_iterator();

				for (std::regex_iterator i = optionsStart; i != optionsEnd; ++i)
				{
					std::smatch match = *i;
					if (match.size() == 4)
					{
						properties.insert_or_assign(match[2], std::make_pair(match[1], match[3]));
					}
				}
			}

			ParseShaderData(options, properties, data);
		}
		else
		{
			shaderCode = shaderBlock;
			return true;
		}

		return true;
	}
}
