#include "HLSLShaderProcessor.h"

#include "Blueberry\Graphics\GraphicsAPI.h"
#include "Blueberry\Tools\StringHelper.h"
#include "Blueberry\Tools\FileHelper.h"

#include "HLSLShaderParser.h"
#include "HLSLShaderCompiler.h"

#include <filesystem>
#include <fstream>

namespace Blueberry
{
	bool HLSLShaderProcessor::Compile(const String& path)
	{
		ShaderCompilationData compilationData = {};
		if (HLSLShaderParser::Parse(path, m_ShaderData, compilationData))
		{
			for (size_t i = 0; i < compilationData.passes.size(); ++i)
			{
				auto& compilationPass = compilationData.passes[i];
				PassData& pass = compilationData.dataPasses[i];
				size_t vertexVariantCount = std::max(static_cast<int>(pow(2, compilationPass.vertexKeywords.size())), 1);
				size_t fragmentVariantCount = std::max(static_cast<int>(pow(2, compilationPass.fragmentKeywords.size())), 1);
				std::unique_ptr<HLSLShaderCompiler> compiler;
				if (GraphicsAPI::GetAPI() == GraphicsAPI::API::DX11)
				{
					compiler = std::make_unique<HLSLShaderCompilerFXC>(compilationPass.shaderCode);
				}
				else
				{
					compiler = std::make_unique<HLSLShaderCompilerDXC>(compilationPass.shaderCode);
				}
				
				if (!compilationPass.vertexEntryPoint.empty())
				{
					compiler->SetKeywords(compilationPass.vertexKeywords);
					pass.SetVertexOffset(static_cast<uint32_t>(m_VariantsData.vertexShaderIndices.size()));

					for (size_t j = 0; j < vertexVariantCount; ++j)
					{
						ByteData data;
						if (!compiler->Compile(compilationPass.vertexEntryPoint.c_str(), HLSLShaderCompilerProfile::Vertex, j, data))
						{
							return false;
						}
						m_VariantsData.vertexShaderIndices.push_back(static_cast<uint32_t>(m_VariantsData.shaders.size()));
						m_VariantsData.shaders.push_back(std::move(data));
					}
				}
				else
				{
					return false;
				}

				if (!compilationPass.geometryEntryPoint.empty())
				{
					pass.SetGeometryOffset(static_cast<uint32_t>(m_VariantsData.geometryShaderIndices.size()));

					ByteData data;
					if (!compiler->Compile(compilationPass.geometryEntryPoint.c_str(), HLSLShaderCompilerProfile::Geometry, 0, data))
					{
						return false;
					}
					m_VariantsData.geometryShaderIndices.push_back(static_cast<uint32_t>(m_VariantsData.shaders.size()));
					m_VariantsData.shaders.push_back(std::move(data));
				}
				else
				{
					m_VariantsData.geometryShaderIndices.push_back(-1);
				}

				if (!compilationPass.fragmentEntryPoint.empty())
				{
					compiler->SetKeywords(compilationPass.fragmentKeywords);
					pass.SetFragmentOffset(static_cast<uint32_t>(m_VariantsData.fragmentShaderIndices.size()));

					for (size_t j = 0; j < fragmentVariantCount; ++j)
					{
						ByteData data;
						if (!compiler->Compile(compilationPass.fragmentEntryPoint.c_str(), HLSLShaderCompilerProfile::Fragment, j, data))
						{
							return false;
						}
						m_VariantsData.fragmentShaderIndices.push_back(static_cast<uint32_t>(m_VariantsData.shaders.size()));
						m_VariantsData.shaders.push_back(std::move(data));
					}
				}
				else
				{
					return false;
				}
			}
			m_ShaderData.SetPasses(compilationData.dataPasses);
		}
		return true;
	}

	void HLSLShaderProcessor::SaveVariants(const String& folderPath)
	{
		std::filesystem::path indexesPath = folderPath;
		indexesPath.append("indexes");

		uint32_t vertexShaderCount = static_cast<uint32_t>(m_VariantsData.vertexShaderIndices.size());
		uint32_t geometryShaderCount = static_cast<uint32_t>(m_VariantsData.geometryShaderIndices.size());
		uint32_t fragmentShaderCount = static_cast<uint32_t>(m_VariantsData.fragmentShaderIndices.size());
		uint32_t blobsCount = static_cast<uint32_t>(m_VariantsData.shaders.size());
		std::ofstream output;
		output.open(indexesPath, std::ofstream::binary);
		output.write(reinterpret_cast<char*>(&vertexShaderCount), sizeof(uint32_t));
		output.write(reinterpret_cast<char*>(m_VariantsData.vertexShaderIndices.data()), sizeof(uint32_t) * vertexShaderCount);
		output.write(reinterpret_cast<char*>(&geometryShaderCount), sizeof(uint32_t));
		output.write(reinterpret_cast<char*>(m_VariantsData.geometryShaderIndices.data()), sizeof(uint32_t) * geometryShaderCount);
		output.write(reinterpret_cast<char*>(&fragmentShaderCount), sizeof(uint32_t));
		output.write(reinterpret_cast<char*>(m_VariantsData.fragmentShaderIndices.data()), sizeof(uint32_t) * fragmentShaderCount);
		output.write(reinterpret_cast<char*>(&blobsCount), sizeof(uint32_t));
		output.close();

		for (size_t i = 0; i < m_VariantsData.shaders.size(); ++i)
		{
			std::filesystem::path path = folderPath;
			path.append(std::to_string(i));
			FileHelper::Save(m_VariantsData.shaders[i], StringHelper::ToString(path));
		}
	}

	bool HLSLShaderProcessor::LoadVariants(const String& folderPath)
	{
		std::filesystem::path indexesPath = folderPath;
		indexesPath.append("indexes");

		if (std::filesystem::exists(indexesPath))
		{
			uint32_t vertexShaderCount;
			uint32_t geometryShaderCount;
			uint32_t fragmentShaderCount;
			uint32_t blobsCount;
			std::ifstream input;
			input.open(indexesPath, std::ofstream::binary);
			input.read(reinterpret_cast<char*>(&vertexShaderCount), sizeof(uint32_t));
			m_VariantsData.vertexShaderIndices.resize(vertexShaderCount);
			input.read(reinterpret_cast<char*>(m_VariantsData.vertexShaderIndices.data()), sizeof(uint32_t) * vertexShaderCount);
			input.read(reinterpret_cast<char*>(&geometryShaderCount), sizeof(uint32_t));
			m_VariantsData.geometryShaderIndices.resize(geometryShaderCount);
			input.read(reinterpret_cast<char*>(m_VariantsData.geometryShaderIndices.data()), sizeof(uint32_t) * geometryShaderCount);
			input.read(reinterpret_cast<char*>(&fragmentShaderCount), sizeof(uint32_t));
			m_VariantsData.fragmentShaderIndices.resize(fragmentShaderCount);
			input.read(reinterpret_cast<char*>(m_VariantsData.fragmentShaderIndices.data()), sizeof(uint32_t) * fragmentShaderCount);
			input.read(reinterpret_cast<char*>(&blobsCount), sizeof(uint32_t));
			input.close();

			for (uint32_t i = 0; i < blobsCount; ++i)
			{
				std::filesystem::path path = folderPath;
				path.append(std::to_string(i));
				String stringPath = StringHelper::ToString(path);
				if (!std::filesystem::exists(path))
				{
					BB_ERROR("Failed to load shader: " << stringPath);
					return false;
				}
				ByteData data = FileHelper::LoadBinary(stringPath);
				m_VariantsData.shaders.push_back(std::move(data));
			}
			return true;
		}
		return false;
	}

	const ShaderData& HLSLShaderProcessor::GetShaderData()
	{
		return m_ShaderData;
	}

	const VariantsData& HLSLShaderProcessor::GetVariantsData()
	{
		return m_VariantsData;
	}
}
