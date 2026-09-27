-- Include once from premake5.lua, then call copychanged { {source, destination}, ... }.
-- Generated builds invoke this file directly, without regenerating the workspace.
-- Workspace tokens become $(SolutionDir) in Visual Studio projects.

local function quote(value)
    assert(type(value) == "string" and not value:find('[\r\n"]'),
        "Copy paths must be strings without quotes or newlines")
    return '"' .. value .. '"'
end

function copychanged(entries)
    assert(#entries > 0, "copychanged requires at least one source/destination pair")
    local command = {
        quote("%{wks.location}/vendor/premake/bin/" .. path.getname(_PREMAKE_COMMAND)),
        quote("--file=%{wks.location}/scripts/CopyChanged.lua"),
        "copychanged",
    }
    for _, entry in ipairs(entries) do
        assert(#entry == 2, "copychanged expects {source, destination} pairs")
        table.insert(command, quote(entry[1]))
        table.insert(command, quote(entry[2]))
    end
    postbuildcommands { table.concat(command, " ") }
end

local function copyFile(source, destination)
    if os.isfile(destination) then
        local equal, err = os.comparefiles(source, destination)
        assert(equal ~= nil, err)
        if equal then
            return
        end
    end
    assert(not os.isdir(destination), "Expected a destination filename: " .. destination)
    assert(os.mkdir(path.getdirectory(destination)))
    local ok, err = os.copyfile(source, destination)
    assert(ok, err)
end

local function copyDirectory(source, destination)
    assert(os.mkdir(destination))
    for _, file in ipairs(os.matchfiles(source .. "/*")) do
        copyFile(file, path.join(destination, path.getname(file)))
    end
    for _, directory in ipairs(os.matchdirs(source .. "/*")) do
        copyDirectory(directory, path.join(destination, path.getname(directory)))
    end
end

newaction {
    trigger = "copychanged",
    description = "Copy source/destination pairs, skipping identical file contents",
    execute = function()
        assert(#_ARGS > 0 and #_ARGS % 2 == 0,
            "Usage: premake5 --file=CopyChanged.lua copychanged SOURCE DESTINATION [...]")
        for index = 1, #_ARGS, 2 do
            -- Relative command-line paths belong to the caller, not this script folder.
            local source = path.getabsolute(_ARGS[index], _WORKING_DIR)
            local destination = path.getabsolute(_ARGS[index + 1], _WORKING_DIR)
            if os.isdir(source) then
                local sourceKey, destinationKey = source, destination
                if os.host() == "windows" then
                    sourceKey, destinationKey = source:lower(), destination:lower()
                end
                assert(destinationKey ~= sourceKey and
                    destinationKey:sub(1, #sourceKey + 1) ~= sourceKey .. "/",
                    "Destination must not be inside its source directory: " .. destination)
                copyDirectory(source, destination)
            elseif os.isfile(source) then
                copyFile(source, destination)
            else
                error("Copy source does not exist: " .. source, 0)
            end
        end
    end,
}
