#include "AssemblyManager.h"

#include "Blueberry\Core\Base.h"
#include "Editor\Path.h"

#include <sstream>
#include <fstream>

namespace Blueberry
{
	static HMODULE s_GameAssembly = nullptr;
	static uint32_t s_ReloadCount = 0;
	static long long s_SourceLastWriteTime = 0;
	
	String GetDebugReleaseFolder()
	{
#if BB_DEBUG
		return "Debug-windows-x86_64";
#else
		return "Release-windows-x86_64";
#endif
	}

	String GetDebugReleaseRuntimeMode()
	{
#if BB_DEBUG
		return "/MDd";
#else
		return "/MD";
#endif
	}

	String GetHotReloadFolder()
	{
		return String("HotReload").append(std::to_string(s_ReloadCount));
	}

	String GetDLLName()
	{
		return GetHotReloadFolder().append("\\GameAssembly");
	}

	String GetAssemblyPath()
	{
		return String(Path::GetAssemblyPath().string());
	}

	void AssemblyManager::Unload()
	{
		if (s_GameAssembly != nullptr)
		{
			FreeLibrary(s_GameAssembly);
			s_GameAssembly = nullptr;
		}
	}

	long long GetLastSourceWriteTime()
	{
		long long result = 0;
		String sourcePath = GetAssemblyPath().append("\\src");
		for (const auto& entry : std::filesystem::directory_iterator(sourcePath))
		{
			auto writeTime = std::chrono::duration_cast<std::chrono::seconds>(std::filesystem::last_write_time(entry.path()).time_since_epoch()).count();
			result = std::max(result, writeTime);
		}
		return result;
	}

	long long GetLastDllWriteTime()
	{
		String dllPath = GetAssemblyPath().append("\\bin\\").append(GetDebugReleaseFolder()).append("\\GameAssembly\\GameAssembly.dll");
		if (std::filesystem::exists(dllPath))
		{
			return std::chrono::duration_cast<std::chrono::seconds>(std::filesystem::last_write_time(dllPath).time_since_epoch()).count();
		}
		return 0;
	}

	void CopyFiles(const String& dllDirectory)
	{
		String originalDllPath = String(dllDirectory).append("GameAssembly.dll");
		String originalPdbPath = String(dllDirectory).append("GameAssembly.pdb");
		if (std::filesystem::exists(originalDllPath) && std::filesystem::exists(originalPdbPath))
		{
			String newPath = String(dllDirectory).append(GetDLLName());
			String hotReloadDirectory = String(dllDirectory).append(GetHotReloadFolder());
			if (!std::filesystem::exists(hotReloadDirectory))
			{
				std::filesystem::create_directory(hotReloadDirectory);
			}
			std::filesystem::copy(originalDllPath, String(newPath).append(".dll"), std::filesystem::copy_options::overwrite_existing);
			std::filesystem::copy(originalPdbPath, String(newPath).append(".pdb"), std::filesystem::copy_options::overwrite_existing);
		}
	}

	void AssemblyManager::Load()
	{
		if (!std::filesystem::exists(GetAssemblyPath()))
		{
			return;
		}

		String dllDirectory = GetAssemblyPath().append("\\bin\\").append(GetDebugReleaseFolder()).append("\\GameAssembly\\");
		String dllPath = String(dllDirectory).append(GetDLLName()).append(".dll");
		
		if (std::filesystem::exists(String(dllDirectory).append("GameAssembly.dll")))
		{
			CopyFiles(dllDirectory);

			using EntryFunc = void(*)();
			s_GameAssembly = LoadLibraryA(dllPath.c_str());
			EntryFunc entryFunc = (EntryFunc)GetProcAddress(s_GameAssembly, "Entry");
			if (entryFunc != nullptr)
			{
				entryFunc();
			}
		}
	}

	String GetMVCSPath()
	{
		String command = "\"C:\\Program Files (x86)\\Microsoft Visual Studio\\Installer\\vswhere.exe\" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath";
		Array<char, 512> buffer;
		String kitsPath;

		FILE* pipe = _popen(command.c_str(), "r");
		if (!pipe)
		{
			return "ERROR";
		}

		while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr)
		{
			kitsPath += buffer.data();
		}

		_pclose(pipe);
		kitsPath.erase(kitsPath.end() - 1, kitsPath.end());
		kitsPath.append("\\VC\\Tools\\MSVC");

		std::vector<String> versions;
		for (const auto& entry : std::filesystem::directory_iterator(kitsPath))
		{
			if (entry.is_directory())
			{
				versions.push_back(String(entry.path().filename().string()));
			}
		}

		if (!versions.empty())
		{
			std::sort(versions.begin(), versions.end(), std::greater<>());
			return kitsPath + "\\" + versions.front();
		}
		return "";
	}

	String GetWindowsSDKIncludePath()
	{
		String kitsPath = "C:\\Program Files (x86)\\Windows Kits\\10\\Include";

		std::vector<String> versions;
		for (const auto& entry : std::filesystem::directory_iterator(kitsPath)) 
		{
			if (entry.is_directory()) 
			{
				versions.push_back(String(entry.path().filename().string()));
			}
		}

		if (!versions.empty())
		{
			std::sort(versions.begin(), versions.end(), std::greater<>());
			return kitsPath + "\\" + versions.front();
		}
		return "";
	}

	String GetWindowsSDKLibPath()
	{
		String kitsPath = "C:\\Program Files (x86)\\Windows Kits\\10\\Lib";

		std::vector<String> versions;
		for (const auto& entry : std::filesystem::directory_iterator(kitsPath))
		{
			if (entry.is_directory())
			{
				versions.push_back(String(entry.path().filename().string()));
			}
		}

		if (!versions.empty())
		{
			std::sort(versions.begin(), versions.end(), std::greater<>());
			return kitsPath + "\\" + versions.front();
		}
		return "";
	}

	String GetEditorPath()
	{
		return String(std::filesystem::current_path().string());
	}

	void CreateCompileAndLinkConfig()
	{
		String msvcPath = GetMVCSPath();
		String windowsSDKIncludePath = GetWindowsSDKIncludePath();
		String windowsSDKLibPath = GetWindowsSDKLibPath();
		String editorPath = GetEditorPath();

		std::stringstream ss;
		ss << ".VSBasePath				= '" << msvcPath << "'\n";
		ss << ".WindowsSDKIncludePath	= '" << windowsSDKIncludePath << "'\n";
		ss << ".WindowsSDKLibPath		= '" << windowsSDKLibPath << "'\n";
		ss << ".EditorBasePath			= '" << editorPath << "'\n";
		ss << "\n";
		ss << "Settings\n";
		ss << "{\n";
		ss << "    .Environment		= { \"PATH = $VSBasePath$..\\..\\..\\..\\Common7\\IDE\\;$VSBasePath$\\bin\\\", \"TMP=C:\\Windows\\Temp\", \"SystemRoot=C:\\Windows\" }\n";
		ss << "}\n";
		ss << "\n";
		ss << ".Compiler				= '$VSBasePath$\\bin\\Hostx64\\x64\\cl.exe'\n";
		ss << ".CompilerOptions		= '\"%1\" /Fo\"%2\" /c /Zi /FS /nologo /std:c++17 /MP " << GetDebugReleaseRuntimeMode() << " /D\"BUILD_DLL\"'\n";
		ss << "\n";
		ss << ".BaseIncludePaths		= ' /I\"./\"'\n";
		ss << "						+ ' /I\"$WindowsSDKIncludePath$\\ucrt\"'\n";
		ss << "						+ ' /I\"$WindowsSDKIncludePath$\\shared\"'\n";
		ss << "						+ ' /I\"$WindowsSDKIncludePath$\\um\"'\n";
		ss << "						+ ' /I\"$WindowsSDKIncludePath$\\winrt\"'\n";
		ss << "						+ ' /I\"$VSBasePath$\\include\"'\n";
		ss << "						+ ' /I\"$EditorBasePath$\\include\"'\n";
		ss << ".CompilerOptions		+ .BaseIncludePaths\n";
		ss << "\n";
		ss << ".Linker					= '$VSBasePath$\\bin\\Hostx64\\x64\\link.exe'\n";
		ss << ".LinkerOptions			= ' /OUT:\"%2\" \"%1\" /NOLOGO /MACHINE:X64 /SUBSYSTEM:WINDOWS /DLL /DEBUG /PDB:\"bin\\" << GetDebugReleaseFolder() << "\\GameAssembly\\GameAssembly.pdb\" /PDBALTPATH:%_PDB%'\n";
		ss << "\n";
		ss << ".LibPaths				= ' /LIBPATH:\"$WindowsSDKLibPath$\\um\\x64\"'\n";
		ss << "						+ ' /LIBPATH:\"$WindowsSDKLibPath$\\ucrt\\x64\"'\n";
		ss << "						+ ' /LIBPATH:\"$VSBasePath$\\lib\\x64\"'\n";
		ss << ".LinkerOptions			+ .LibPaths\n";
		ss << "\n";
		ss << "ObjectList( 'GameAssembly-Obj' )\n";
		ss << "{\n";
		ss << "    .CompilerInputPath	= 'src\\'\n";
		ss << "    .CompilerOutputPath	= 'bin-int\\" << GetDebugReleaseFolder() << "\\GameAssembly\\'\n";
		ss << "}\n";
		ss << "\n";
		ss << "DLL( 'GameAssembly' )\n";
		ss << "{\n";
		ss << "    .Libraries			= { \"GameAssembly-Obj\" }\n";
		ss << "    .LinkerOutput		= 'bin\\" << GetDebugReleaseFolder() << "\\GameAssembly\\GameAssembly.dll'\n";
		ss << "    .LinkerOptions		+ ' uuid.lib'\n";
		ss << "						+ ' $EditorBasePath$\\BlueberryEditor.lib'\n";
		ss << "}\n";
		ss << "\n";
		ss << "Alias( 'all' ) { .Targets = { 'GameAssembly' } }";

		String path = GetAssemblyPath().append("\\fbuild.bff");
		std::ofstream output;
		output.open(path.c_str(), std::ofstream::binary);
		output << ss.rdbuf();
		output.close();
	}

	bool CompileAndLinkProject()
	{
		String command = String("\"").append(GetEditorPath()).append("\\FBuild.exe\"");
		long long writeTime = GetLastDllWriteTime();

		STARTUPINFOA si = { sizeof(si) };
		PROCESS_INFORMATION pi;

		if (!CreateProcessA(nullptr, command.data(), nullptr, nullptr, false, 0, nullptr, GetAssemblyPath().c_str(), &si, &pi))
		{
			DWORD dwErrorCode = GetLastError();
			return false;
		}

		WaitForSingleObject(pi.hProcess, INFINITE);

		return GetLastDllWriteTime() > writeTime;
	}

	bool AssemblyManager::Build(const bool& incrementCount)
	{
		if (!std::filesystem::exists(GetAssemblyPath()))
		{
			return false;
		}
		long long lastWriteTime = GetLastSourceWriteTime();
		if (lastWriteTime <= s_SourceLastWriteTime)
		{
			return false;
		}
		else
		{
			s_SourceLastWriteTime = lastWriteTime;
		}
		CreateCompileAndLinkConfig();
		if (CompileAndLinkProject())
		{
			if (incrementCount)
			{
				++s_ReloadCount;
			}
			return true;
		}
		return false;
	}
}
