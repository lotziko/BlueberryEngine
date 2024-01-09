#pragma once
#include "Blueberry\Core\Guid.h"

namespace Blueberry
{
	class AssetLoader
	{
	public:
		static void Initialize(AssetLoader* loader);
		static Object* Load(const Guid& guid);
		static Object* Load(const std::string& path);

	protected:
		virtual Object* LoadImpl(const Guid& guid) = 0;
		virtual Object* LoadImpl(const std::string& path) = 0;

	private:
		static inline AssetLoader* s_Instance = nullptr;
	};
}