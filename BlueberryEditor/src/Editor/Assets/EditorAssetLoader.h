#pragma once
#include "Blueberry\Assets\AssetLoader.h"

namespace Blueberry
{
	class EditorAssetLoader final : public AssetLoader
	{
	public:
		EditorAssetLoader() = default;

	protected:
		virtual Object* LoadImpl(const Guid& guid, FileId fileId) override;
		virtual Object* LoadImpl(const String& path, void* args) override;

	private:
		Dictionary<String, Object*> m_LoadedAssets;
	};
}