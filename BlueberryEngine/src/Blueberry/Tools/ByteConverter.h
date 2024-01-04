#pragma once

namespace Blueberry
{
	class ByteConverter
	{
	public:
		template<class Type>
		static void HexStringToBytes(const char* hexstr, Type* dst, size_t length);
		template<class Type>
		static void BytesToHexString(const Type* src, char* hexstr, size_t length);
	};

	template<class Type>
	inline void ByteConverter::HexStringToBytes(const char* hexstr, Type* dst, size_t length)
	{
		static const uint_fast8_t LOOKUP[256] = {
			0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
			0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
			0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
			0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
			0x00, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
			0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
			0x00, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f };

		uint_fast8_t* ptr = reinterpret_cast<uint_fast8_t*>(dst);

		for (size_t i = 0; i < length; i += 2)
		{
			*ptr = LOOKUP[hexstr[i]] << 4 |
				LOOKUP[hexstr[i + 1]];
			ptr++;
		}
	}

	template<class Type>
	inline void ByteConverter::BytesToHexString(const Type* src, char* hexstr, size_t length)
	{
		static const uint_fast8_t LOOKUP[] = "0123456789abcdef";
		const uint_fast8_t* ptr = reinterpret_cast<const uint_fast8_t*>(src);

		for (size_t i = 0; i < length; i++)
		{
			uint_fast8_t c = *ptr++;
			*hexstr++ = LOOKUP[c >> 4];
			*hexstr++ = LOOKUP[c & 0x0f];
		}
	}
}