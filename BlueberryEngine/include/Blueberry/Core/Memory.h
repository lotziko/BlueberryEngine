#pragma once

#include "Blueberry\Core\Base.h"

#include <array>
#include <vector>
#include <string>
#include <map>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <set>

namespace Blueberry
{
	class BB_API Allocator
	{
		struct BB_API Initializer
		{
			Initializer();
			~Initializer();
		};

	public:
		static void InitializeThread();
		static void ShutdownThread();

		static void* Allocate(const size_t& size);
		static void Free(void* ptr);

	private:
		static inline Initializer s_Initializer = {};
	};

	// Based on Jolt STLAllocator
	template<typename T>
	class STLAllocator
	{
	public:
		using value_type = T;

		/// Pointer to type
		using pointer = T*;
		using const_pointer = const T*;

		/// Reference to type.
		/// Can be removed in C++20.
		using reference = T&;
		using const_reference = const T&;

		using size_type = size_t;
		using difference_type = ptrdiff_t;

		inline STLAllocator() = default;

		template <typename T2>
		inline STLAllocator(const STLAllocator<T2>&) { }

		inline pointer allocate(size_type n)
		{
			return static_cast<pointer>(Allocator::Allocate(n * sizeof(value_type)));
		}

		inline void	deallocate(pointer p, size_type n)
		{
			Allocator::Free(p);
		}

		inline bool	operator == (const STLAllocator<T> &) const
		{
			return true;
		}

		inline bool	operator != (const STLAllocator<T> &) const
		{
			return false;
		}

		template <typename T2>
		struct rebind
		{
			using other = STLAllocator<T2>;
		};
	};

	template <class T, size_t Size> using Array = std::array<T, Size>;
	template <class T> using List = std::vector<T, STLAllocator<T>>;
	template <class T> using Stack = std::stack<T, STLAllocator<T>>;
	template <class Key, class T, class Hash = std::hash<Key>, class KeyEqual = std::equal_to<Key>> using Dictionary = std::unordered_map<Key, T, Hash, KeyEqual, STLAllocator<std::pair<const Key, T>>>;
	template <class T, class Hash = std::hash<T>, class KeyEqual = std::equal_to<T>> using HashSet = std::unordered_set<T, Hash, KeyEqual, STLAllocator<T>>;
	using String = std::basic_string<char, std::char_traits<char>, STLAllocator<char>>;
	using WString = std::basic_string<wchar_t, std::char_traits<wchar_t>, STLAllocator<wchar_t>>;
}

#define BB_OVERRIDE_NEW_DELETE													\
inline void* operator new(std::size_t size) { return BB_MALLOC(size); }			\
inline void* operator new[](std::size_t size) { return BB_MALLOC(size); }		\
inline void operator delete(void* ptr) { BB_FREE(ptr); }						\
inline void operator delete[](void* ptr) { BB_FREE(ptr); }						\

#define BB_INITIALIZE_ALLOCATOR_THREAD() Blueberry::Allocator::InitializeThread()
#define BB_MALLOC(size) Blueberry::Allocator::Allocate(size)
#define BB_MALLOC_ARRAY(type, count) static_cast<type*>(Blueberry::Allocator::Allocate(sizeof(type) * count))
#define BB_FREE(ptr) Blueberry::Allocator::Free(ptr)
#define BB_SHUTDOWN_ALLOCATOR_THREAD() Blueberry::Allocator::ShutdownThread()