#include "Blueberry\Graphics\Enums.h"

namespace Blueberry
{
	TextureUsageFlags operator|(const TextureUsageFlags& lhs, const TextureUsageFlags& rhs)
	{
		return static_cast<TextureUsageFlags>(static_cast<int>(lhs) | static_cast<int>(rhs));
	}

	TextureUsageFlags operator&(const TextureUsageFlags& lhs, const TextureUsageFlags& rhs)
	{
		return static_cast<TextureUsageFlags>(static_cast<int>(lhs) & static_cast<int>(rhs));
	}

	BufferUsageFlags operator|(const BufferUsageFlags& lhs, const BufferUsageFlags& rhs)
	{
		return static_cast<BufferUsageFlags>(static_cast<int>(lhs) | static_cast<int>(rhs));
	}

	BufferUsageFlags operator&(const BufferUsageFlags& lhs, const BufferUsageFlags& rhs)
	{
		return static_cast<BufferUsageFlags>(static_cast<int>(lhs) & static_cast<int>(rhs));
	}
}