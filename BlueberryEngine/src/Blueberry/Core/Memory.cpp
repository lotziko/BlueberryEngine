#include "Blueberry\Core\Memory.h"

#include <rpmalloc\rpmalloc.h>

namespace Blueberry
{
	Allocator::Initializer::Initializer()
	{
		rpmalloc_initialize();
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

	void* Allocator::Allocate(const size_t& size)
	{
		return rpmalloc(size);
	}

	void Allocator::Free(void* ptr)
	{
		rpfree(ptr);
	}
}
