#pragma once
#include "Blueberry\Assets\AssetLoader.h"

namespace Blueberry
{
	class RuntimeAssetLoader final : public AssetLoader
	{
	public:
		RuntimeAssetLoader() = default;

	protected:
		virtual Object* LoadImpl(const Guid& guid, FileId fileId) override;
		virtual Object* LoadImpl(const String& path, void* args) override;

	private:
		Dictionary<String, Object*> m_LoadedAssets;
	};
}