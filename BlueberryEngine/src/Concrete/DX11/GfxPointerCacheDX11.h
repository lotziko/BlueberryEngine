#pragma once

namespace Blueberry
{
	template <typename T>
	class GfxPointerCacheDX11
	{
	public:
		uint32_t Allocate(T* ptr)
		{
			uint32_t index = static_cast<uint32_t>(s_Pointers.size());
			s_Pointers.push_back(ptr);
			return index;
		}

		void Deallocate(const uint32_t& index)
		{
			if (index >= 0 && index < s_Pointers.size())
			{
				s_Pointers[index] = nullptr;
			}
		}

		T* Get(const uint32_t& index) const
		{
			return s_Pointers[index];
		}

	private:
		List<T*> s_Pointers;
	};
}