#pragma once

namespace Blueberry
{
	class ShaderProcessor
	{
	public:
		static void* Compile(const std::string& path, const char* entryPoint, const char* model, const std::string& blobPath);
		static void* Load(const std::string& path);
	};
}