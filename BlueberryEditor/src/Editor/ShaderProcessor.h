#pragma once

namespace Blueberry
{
	class ShaderProcessor
	{
	public:
		void* Compile(const std::string& path, const char* entryPoint, const char* model, const std::string& blobPath);
		void* Load(const std::string& path);
	};
}