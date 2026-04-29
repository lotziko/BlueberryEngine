#include "Blueberry\Tools\CompressionHelper.h"

#include <lz4\lz4hc.h>

namespace Blueberry
{
	void CompressionHelper::Decompress(const List<uint8_t>& data, List<uint8_t>& decompressedData)
	{
		const char* compressedData = reinterpret_cast<const char*>(data.data());
		LZ4_decompress_safe(compressedData, reinterpret_cast<char*>(decompressedData.data()), static_cast<int>(data.size()), static_cast<int>(decompressedData.size()));
	}

	List<uint8_t> CompressionHelper::Decompress(const List<uint8_t>& data, size_t originalSize)
	{
		List<uint8_t> result(originalSize);
		const char* compressedData = reinterpret_cast<const char*>(data.data());
		char* decompressedData = reinterpret_cast<char*>(result.data());
		LZ4_decompress_safe(compressedData, decompressedData, static_cast<int>(data.size()), static_cast<int>(originalSize));
		return result;
	}

	List<uint8_t> CompressionHelper::Compress(const List<uint8_t>& data)
	{
		List<uint8_t> result;
		const char* srcData = reinterpret_cast<const char*>(data.data());
		int srcSize = static_cast<int>(data.size());
		int maxDstSize = LZ4_compressBound(srcSize);
		result.resize(maxDstSize);
		char* compressedData = reinterpret_cast<char*>(result.data());
		int compressedSize = LZ4_compress_HC(srcData, compressedData, srcSize, maxDstSize, 9);
		result.resize(compressedSize);
		return result;
	}

	List<uint8_t> CompressionHelper::Compress(const String& data)
	{
		List<uint8_t> result;
		const char* srcData = data.data();
		int srcSize = static_cast<int>(data.size());
		int maxDstSize = LZ4_compressBound(srcSize);
		result.resize(maxDstSize);
		char* compressedData = reinterpret_cast<char*>(result.data());
		int compressedSize = LZ4_compress_HC(srcData, compressedData, srcSize, maxDstSize, 9);
		result.resize(compressedSize);
		return result;
	}
}