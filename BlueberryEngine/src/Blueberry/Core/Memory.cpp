#include "Blueberry\Core\Memory.h"

#include <rpmalloc\rpmalloc.h>

namespace Blueberry
{
	Allocator::Initializer::Initializer()
	{
		if (!s_IsInitialized)
		{
			rpmalloc_initialize();
			s_IsInitialized = true;
		}
	}

	Allocator::Initializer::~Initializer()
	{
		//rpmalloc_finalize();
	}

	void Allocator::InitializeThread()
	{
		rpmalloc_thread_initialize();
	}

	void Allocator::ShutdownThread()
	{
		rpmalloc_thread_finalize(0);
	}

	void* Allocator::Allocate(size_t size)
	{
		return rpmalloc(size);
	}

	void Allocator::Free(void* ptr)
	{
		rpfree(ptr);
	}
}
