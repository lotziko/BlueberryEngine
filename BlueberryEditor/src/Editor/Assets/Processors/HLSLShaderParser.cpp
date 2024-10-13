#include "bbpch.h"
#include "HLSLShaderParser.h"

#include "Blueberry\Tools\FileHelper.h"
#include "Blueberry\Tools\StringHelper.h"
#include "Editor\Assets\Processors\HLSLShaderProcessor.h"

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

	bool ParseBlock(std::string& text, const std::string& name, const std::string& begin, const std::string& end, std::string& result)
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
		else if (name == "None")
		{
			return CullMode::None;
		}
		return CullMode::None;
	}

	BlendMode ParseBlendMode(const std::string& name)
	{
		if (name == "Zero")
		{
			return BlendMode::Zero;
		}
		else if (name == "One")
		{
			return BlendMode::One;
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

	ZTest ParseZTest(const std::string& name)
	{
		if (name == "Never")
		{
			return ZTest::Never;
		}
		else if (name == "Less")
		{
			return ZTest::Less;
		}
		else if (name == "Equal")
		{
			return ZTest::Equal;
		}
		else if (name == "LessEqual")
		{
			return ZTest::LessEqual;
		}
		else if (name == "Greater")
		{
			return ZTest::Greater;
		}
		else if (name == "NotEqual")
		{
			return ZTest::NotEqual;
		}
		else if (name == "GreaterEqual")
		{
			return ZTest::GreaterEqual;
		}
		else if (name == "Always")
		{
			return ZTest::Always;
		}
		return ZTest::LessEqual;
	}

	ZWrite ParseZWrite(const std::string& name)
	{
		if (name == "Off")
		{
			return ZWrite::Off;
		}
		return ZWrite::On;
	}

	void ParseProperties(const std::string& propertiesBlock, ShaderData& data)
	{
		std::regex propertyRegex("([\\w-]+)\\s*([\\w-]+)\\s*[\\=]\\s*(.*)[\r?\n]");
		auto propertiesStart = std::sregex_iterator(propertiesBlock.begin(), propertiesBlock.end(), propertyRegex);
		auto propertiesEnd = std::sregex_iterator();

		std::vector<DataPtr<TextureParameterData>> textureParameters;
		for (std::regex_iterator i = propertiesStart; i != propertiesEnd; ++i)
		{
			std::smatch match = *i;
			if (match.size() == 4)
			{
				std::string type = match[1];
				std::string name = match[2];
				std::string value = match[3];

				if (type == "Texture2D")
				{
					std::string defaultTextureName = value;
					defaultTextureName.erase(std::remove(defaultTextureName.begin(), defaultTextureName.end(), '\"'), defaultTextureName.end());
				
					TextureParameterData* parameter = new TextureParameterData();
					parameter->SetName(name);
					parameter->SetDefaultTextureName(defaultTextureName);
					textureParameters.emplace_back(DataPtr<TextureParameterData>(parameter));
				}
			}
		}
		data.SetTextureParameters(textureParameters);
	}

	void ParseRenderingParameters(const std::string& passBlock, PassData& passData)
	{
		std::smatch match;
		std::regex cullRegex("Cull\\s*([\\w-]+)[\r?\n]");
		if (std::regex_search(passBlock, match, cullRegex))
		{
			passData.SetCullMode(ParseCullMode(match[1]));
		}
		
		std::regex blendLongRegex("Blend\\s([\\w-]+)\\s([\\w-]+)\\s([\\w-]+)\\s([\\w-]+)[\r?\n]");
		std::regex blendShortRegex("Blend\\s*([\\w-]+)\\s*([\\w-]+)[\r?\n]");
		if (std::regex_search(passBlock, match, blendLongRegex))
		{
			// SrcColor SrcAlpha DstColor DstAlpha
			passData.SetBlendSrc(ParseBlendMode(match[1]), ParseBlendMode(match[3]));
			passData.SetBlendDst(ParseBlendMode(match[2]), ParseBlendMode(match[4]));
		}
		else if (std::regex_search(passBlock, match, blendShortRegex))
		{
			passData.SetBlendSrc(ParseBlendMode(match[1]));
			passData.SetBlendDst(ParseBlendMode(match[2]));
		}

		std::regex zTestRegex("ZTest\\s*([\\w-]+)[\r?\n]");
		if (std::regex_search(passBlock, match, zTestRegex))
		{
			passData.SetZTest(ParseZTest(match[1]));
		}

		std::regex zWriteRegex("ZWrite\\s*([\\w-]+)[\r?\n]");
		if (std::regex_search(passBlock, match, zWriteRegex))
		{
			passData.SetZWrite(ParseZWrite(match[1]));
		}
	}

	void ParsePragmas(const std::string& codeBlock, PassData& passData, ShaderCompilationData::Pass& compilationPass)
	{
		// TODO do real preprocessing
		std::smatch match;
		std::regex vertexEntryPointRegex("#pragma\\s*vertex\\s*([\\w-]+)[\r?\n]");
		if (std::regex_search(codeBlock, match, vertexEntryPointRegex))
		{
			compilationPass.vertexEntryPoint = match[1];
		}

		std::regex geometryEntryPointRegex("#pragma\\s*geometry\\s*([\\w-]+)[\r?\n]");
		if (std::regex_search(codeBlock, match, geometryEntryPointRegex))
		{
			compilationPass.geometryEntryPoint = match[1];
		}

		std::regex fragmentEntryPointRegex("#pragma\\s*fragment\\s*([\\w-]+)[\r?\n]");
		if (std::regex_search(codeBlock, match, fragmentEntryPointRegex))
		{
			compilationPass.fragmentEntryPoint = match[1];
		}

		std::regex parameterRegex("#pragma\\s*keyword_([\\w-]+)\\s*(.*)");
		auto parametersStart = std::sregex_iterator(codeBlock.begin(), codeBlock.end(), parameterRegex);
		auto parametersEnd = std::sregex_iterator();

		for (std::regex_iterator i = parametersStart; i != parametersEnd; ++i)
		{
			std::smatch match = *i;
			std::string type = match[1];
			std::string keywords = match[2];

			if (type == "local_vertex")
			{
				StringHelper::Split(keywords.c_str(), ' ', compilationPass.vertexKeywords);
			}
			else if (type == "local_fragment")
			{
				StringHelper::Split(keywords.c_str(), ' ', compilationPass.fragmentKeywords);
			}
		}
		passData.SetVertexKeywords(compilationPass.vertexKeywords);
		passData.SetFragmentKeywords(compilationPass.fragmentKeywords);
	}

	bool HLSLShaderParser::Parse(const std::string& path, ShaderData& shaderData, ShaderCompilationData& compilationData)
	{
		std::string shader;
		FileHelper::Load(shader, path);

		std::string shaderBlock;
		if (ParseBlock(shader, "Shader", "{", "}", shaderBlock))
		{
			std::string propertiesBlock;
			if (ParseBlock(shaderBlock, "Properties", "{", "}", propertiesBlock))
			{
				ParseProperties(propertiesBlock, shaderData);
			}

			std::string passBlock;
			while (ParseBlock(shaderBlock, "Pass", "{", "}", passBlock))
			{
				// Replace "Pass" keyword to parse the next one
				size_t passOffset = shaderBlock.find("Pass", 0);
				shaderBlock.replace(passOffset, strlen("Pass"), " ");

				ShaderCompilationData::Pass compilationPass = {};
				PassData passData = {};

				std::string codeBlock;
				if (!ParseBlock(passBlock, "", "HLSLBEGIN", "HLSLEND", codeBlock))
				{
					return false;
				}

				ParseRenderingParameters(passBlock, passData);
				ParsePragmas(codeBlock, passData, compilationPass);

				size_t offset = 0;
				while ((offset = codeBlock.find("#pragma")) != std::string::npos)
				{
					size_t end = codeBlock.find("\n", offset);
					codeBlock.replace(offset, end - offset, " ");
				}

				compilationPass.shaderCode = codeBlock;
				compilationData.passes.emplace_back(compilationPass);
				compilationData.dataPasses.emplace_back(new PassData(passData));
			}
			return true;
		}

		return false;
	}
}
