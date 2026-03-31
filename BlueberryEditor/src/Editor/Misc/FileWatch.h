#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	enum class FileOperation
	{
		Add,
		Remove,
		Modify,
		Rename
	};

	struct FileOperationInfo
	{
		FileOperation operation;
		String path;
		String renamePath;
	};

	class FileWatch
	{
	public:
		virtual const List<FileOperationInfo>& GetFileOperations() = 0;
		virtual void ClearFileOperations() = 0;

		static FileWatch* Create(const String& path);
	};
}