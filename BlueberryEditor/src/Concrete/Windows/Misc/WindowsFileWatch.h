#pragma once

#include "Editor\Misc\FileWatch.h"

namespace Blueberry
{
	class WindowsFileWatch : public FileWatch
	{
	public:
		WindowsFileWatch(const String& directory);
		~WindowsFileWatch();

		virtual const List<FileOperationInfo>& GetFileOperations() final;
		virtual void ClearFileOperations() final;

	private:
		static DWORD WINAPI WatchDirectory(LPVOID param);

	private:
		List<FileOperationInfo> m_FileOperations;
		HANDLE m_FileHandle = 0;
		HANDLE m_WatchThreadHandle = 0;
	};
}