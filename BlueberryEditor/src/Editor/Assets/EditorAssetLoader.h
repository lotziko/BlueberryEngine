#pragma once
#include "Blueberry\Assets\AssetLoader.h"

namespace Blueberry
{
	class EditorAssetLoader final : public AssetLoader
	{
	protected:
		virtual void LoadImpl(const Guid& guid) override;
		virtual Object* LoadImpl(const Guid& guid, const FileId& fileId) override;
		virtual Object* LoadImpl(const std::string& path) override;
	};
}