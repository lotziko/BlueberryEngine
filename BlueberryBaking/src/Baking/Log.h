#pragma once

#include <optix.h>
#include <cuda_runtime.h>

#include <iostream>
#include <sstream>
#include <string>

namespace Blueberry
{
	class Exception : public std::runtime_error
	{
	public:
		Exception(const char* msg)
			: std::runtime_error(msg)
		{ }

		Exception(OptixResult res, const char* msg)
			: std::runtime_error(createMessage(res, msg).c_str())
		{ }

	private:
		std::string createMessage(OptixResult res, const char* msg)
		{
			std::ostringstream out;
			out << optixGetErrorName(res) << ": " << msg;
			return out.str();
		}
	};

#define OPTIX_CHECK( call )                                                    \
    do                                                                         \
    {                                                                          \
        OptixResult res = call;                                                \
        if( res != OPTIX_SUCCESS )                                             \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << "Optix call '" << #call << "' failed: " __FILE__ ":"         \
               << __LINE__ << ")\n";                                           \
			std::cout << ss.str().c_str() << std::endl;				           \
			throw Exception( ss.str().c_str() );						       \
        }                                                                      \
    } while( 0 )

#define OPTIX_CHECK_LOG( call )                                                \
    do                                                                         \
    {                                                                          \
        OptixResult res = call;                                                \
        const size_t sizeof_log_returned = sizeofLog;                          \
        sizeofLog = sizeof( log ); /* reset sizeof_log for future calls */     \
        if( res != OPTIX_SUCCESS )                                             \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << "Optix call '" << #call << "' failed: " __FILE__ ":"         \
               << __LINE__ << ")\nLog:\n" << log                               \
               << ( sizeof_log_returned > sizeof( log ) ? "<TRUNCATED>" : "" ) \
               << "\n";                                                        \
			std::cout << ss.str().c_str() << std::endl;				           \
			throw Exception( ss.str().c_str() );						       \
        }                                                                      \
    } while( 0 )

#define CUDA_CHECK( call )                                                     \
    do                                                                         \
    {                                                                          \
        cudaError_t error = call;                                              \
        if( error != cudaSuccess )                                             \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << "CUDA call (" << #call << " ) failed with error: '"          \
               << cudaGetErrorString( error )                                  \
               << "' (" __FILE__ << ":" << __LINE__ << ")\n";                  \
			std::cout << ss.str().c_str() << std::endl;				           \
			throw Exception( ss.str().c_str() );						       \
        }                                                                      \
    } while( 0 )

#define CUDA_CHECK_RESULT( call )                                              \
    do                                                                         \
    {                                                                          \
        CUresult error = call;								                   \
		const char *err_string;												   \
        if( error != CUDA_SUCCESS )                                            \
        {                                                                      \
            std::stringstream ss;                                              \
			cuGetErrorString( error, &err_string );							\
            ss << "CUDA call (" << #call << " ) failed with error: '"          \
               << err_string													\
               << "' (" __FILE__ << ":" << __LINE__ << ")\n";                  \
			std::cout << ss.str().c_str() << std::endl;				           \
			throw Exception( ss.str().c_str() );						       \
        }                                                                      \
    } while( 0 )

#define CUDA_SYNC_CHECK()                                                      \
    do                                                                         \
    {                                                                          \
        cudaDeviceSynchronize();                                               \
        cudaError_t error = cudaGetLastError();                                \
        if( error != cudaSuccess )                                             \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << "CUDA error on synchronize with error '"                     \
               << cudaGetErrorString( error )                                  \
               << "' (" __FILE__ << ":" << __LINE__ << ")\n";                  \
			std::cout << ss.str().c_str() << std::endl;				           \
			throw Exception( ss.str().c_str() );						       \
        }                                                                      \
    } while( 0 )
}