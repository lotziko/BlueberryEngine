#pragma once

#include "Blueberry\Core\Guid.h"

namespace Blueberry
{
	class Object;

	class AssetLoader
	{
	public:
		static void Initialize(AssetLoader* loader);
		static void Load(const Guid& guid);
		static Object* Load(const Guid& guid, const FileId& fileId);
		static Object* Load(const String& path, void* args = nullptr);

	protected:
		virtual void LoadImpl(const Guid& guid) = 0;
		virtual Object* LoadImpl(const Guid& guid, const FileId& fileId) = 0;
		virtual Object* LoadImpl(const String& path, void* args) = 0;

	private:
		static inline AssetLoader* s_Instance = nullptr;
	};
}