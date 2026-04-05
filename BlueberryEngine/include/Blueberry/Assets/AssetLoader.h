#pragma once

#include "Blueberry\Core\Guid.h"

namespace Blueberry
{
	class Object;

	class AssetLoader
	{
	public:
		static void Initialize(AssetLoader* loader);
		static Object* Load(const Guid& guid, FileId fileId);
		static Object* Load(const String& path, void* args = nullptr);
		static AssetLoader* GetInstance();

	protected:
		virtual Object* LoadImpl(const Guid& guid, FileId fileId) = 0;
		virtual Object* LoadImpl(const String& path, void* args) = 0;

	private:
		static AssetLoader* s_Instance;
	};
}