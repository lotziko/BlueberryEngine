#pragma once

#include "Log.h"
#include <optix.h>
// common std stuff
#include <vector>

namespace Blueberry
{
	//https://github.com/ingowald/optix7course/blob/1032df3a87d1fa0d935071183d9c57aa09ef99d6/example02_pipelineAndRayGen/CUDABuffer.h
/*! simple wrapper for creating, and managing a device-side CUDA
	 buffer */
	struct CUDABuffer {

		//! re-size buffer to given number of bytes
		void resize(size_t size)
		{
			if (data != 0) free();
			alloc(size);
		}

		//! allocate to given number of bytes
		void alloc(size_t size)
		{
			this->sizeInBytes = size;
			CUDA_CHECK(cudaMalloc((void**)&data, sizeInBytes));
		}

		//! free allocated memory
		void free()
		{
			CUDA_CHECK(cudaFree((void *)data));
			sizeInBytes = 0;
		}

		template<typename T>
		void alloc_and_upload(const std::vector<T> &vt)
		{
			alloc(vt.size() * sizeof(T));
			upload((const T*)vt.data(), vt.size());
		}

		template<typename T>
		void alloc_and_upload(const T *t, size_t count)
		{
			alloc(count * sizeof(T));
			upload(t, count);
		}

		template<typename T>
		void upload(const T *t, size_t count)
		{
			CUDA_CHECK(cudaMemcpy((void *)data, (void *)t,
				count * sizeof(T), cudaMemcpyHostToDevice));
		}

		template<typename T>
		void download(T *t, size_t count)
		{
			CUDA_CHECK(cudaMemcpy((void *)t, (void *)data,
				count * sizeof(T), cudaMemcpyDeviceToHost));
		}

		void clear()
		{
			CUDA_CHECK(cudaMemset((void *)data, 0, sizeInBytes));
		}

		size_t sizeInBytes{ 0 };
		CUdeviceptr data = 0;
	};
}