#include "WindowsFileWatch.h"

#include "Blueberry\Tools\StringConverter.h"

namespace Blueberry
{
	WindowsFileWatch::WindowsFileWatch(const String& path)
	{
		WString wpath = StringConverter::StringToWide(path);

		m_FileHandle = CreateFile(wpath.c_str(),
			FILE_LIST_DIRECTORY,
			FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
			NULL,
			OPEN_EXISTING,
			FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OVERLAPPED,
			NULL);
		
		m_WatchThreadHandle = CreateThread(NULL, 0, WatchDirectory, this, 0, NULL);
	}

	WindowsFileWatch::~WindowsFileWatch()
	{
		CloseHandle(m_WatchThreadHandle);
	}

	const List<FileOperationInfo>& WindowsFileWatch::GetFileOperations()
	{
		return m_FileOperations;
	}

	void WindowsFileWatch::ClearFileOperations()
	{
		m_FileOperations.clear();
	}

	DWORD WINAPI WindowsFileWatch::WatchDirectory(LPVOID param)
	{
		BB_INITIALIZE_ALLOCATOR_THREAD();
		WindowsFileWatch* self = static_cast<WindowsFileWatch*>(param);

		OVERLAPPED overlapped;
		overlapped.hEvent = CreateEvent(NULL, FALSE, 0, NULL);

		String oldName = {};
		uint8_t change_buf[1024];
		BOOL success = ReadDirectoryChangesW(
			self->m_FileHandle, change_buf, 1024, TRUE,
			FILE_NOTIFY_CHANGE_FILE_NAME |
			FILE_NOTIFY_CHANGE_DIR_NAME |
			FILE_NOTIFY_CHANGE_LAST_WRITE,
			NULL, &overlapped, NULL);

		while (true)
		{
			DWORD result = overlapped.hEvent != 0 ? WaitForSingleObject(overlapped.hEvent, INFINITE) : WAIT_FAILED;

			if (result == WAIT_OBJECT_0)
			{
				DWORD bytes_transferred;
				GetOverlappedResult(self->m_FileHandle, &overlapped, &bytes_transferred, FALSE);

				FILE_NOTIFY_INFORMATION* event = (FILE_NOTIFY_INFORMATION*)change_buf;

				for (;;)
				{
					DWORD name_len = event->FileNameLength / sizeof(wchar_t);
					String name = StringConverter::WideToString(WString(event->FileName, name_len));

					switch (event->Action)
					{
					case FILE_ACTION_ADDED:
					{
						FileOperationInfo info = {};
						info.operation = FileOperation::Add;
						info.path = name;
						self->m_FileOperations.push_back(info);
					}
					break;
					case FILE_ACTION_REMOVED:
					{
						FileOperationInfo info = {};
						info.operation = FileOperation::Remove;
						info.path = name;
						self->m_FileOperations.push_back(info);
					}
					break;
					case FILE_ACTION_MODIFIED:
					{
						FileOperationInfo info = {};
						info.operation = FileOperation::Modify;
						info.path = name;
						self->m_FileOperations.push_back(info);
					}
					break;
					case FILE_ACTION_RENAMED_OLD_NAME:
					{
						oldName = name;
					}
					break;
					case FILE_ACTION_RENAMED_NEW_NAME:
					{
						FileOperationInfo info = {};
						info.operation = FileOperation::Rename;
						info.path = name;
						info.renamePath = oldName;
						self->m_FileOperations.push_back(info);
					}
					break;
					}

					// Are there more events to handle?
					if (event->NextEntryOffset)
					{
						*((uint8_t**)&event) += event->NextEntryOffset;
					}
					else
					{
						break;
					}
				}

				// Queue the next event
				BOOL success = ReadDirectoryChangesW(
					self->m_FileHandle, change_buf, 1024, TRUE,
					FILE_NOTIFY_CHANGE_FILE_NAME |
					FILE_NOTIFY_CHANGE_DIR_NAME |
					FILE_NOTIFY_CHANGE_LAST_WRITE,
					NULL, &overlapped, NULL);
			}
		}
		BB_SHUTDOWN_ALLOCATOR_THREAD();
		return 0;
	}
}