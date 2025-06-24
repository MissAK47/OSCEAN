#include "common_utils/utilities/filesystem_utils.h"
#include "common_utils/utilities/logging_utils.h"
#include <filesystem>
#include <fstream> // For createTempFile to close the file descriptor, though not strictly for manipulation logic itself
#include <system_error>
#include <vector> // For createTempFile pattern
#include <cstdio> // For std::rename in one of the original implementations, mkstemp, mkdtemp

#ifdef _WIN32
#include <windows.h> // For GetTempFileNameA, MAX_PATH
#else
#include <unistd.h> // For close, unlink (though fs::remove is preferred)
// mkstemp, mkdtemp are in cstdlib/stdlib.h, often included via iostream or other headers but good to be aware
#endif

namespace oscean {
namespace common_utils {

namespace fs = std::filesystem;

// Static helper function, similar to the one in other filesystem_*.cpp files
static void logFilesystemError(const std::string& operation, const fs::path& path, const std::error_code& ec) {
    std::string path_str = path.string();
    LOG_ERROR("Filesystem error during '{}' on path '{}': {} ({})", 
              operation, path_str, ec.message(), ec.value());
}

// ensureDirectoryExists is in filesystem_io_utils.cpp. Calls to it from here are valid.
// isDirectory is in filesystem_query_utils.cpp. Calls to it from here are valid.

bool FilesystemUtils::createDirectory(const fs::path& path, bool recursive) {
    try {
        std::error_code ec;
        bool success = false;
        if (recursive) {
            success = fs::create_directories(path, ec);
        } else {
            // Check if parent directory exists before creating non-recursively
            fs::path parent = path.parent_path();
            if (!parent.empty()) {
                std::error_code ec_parent_exists;
                if (!fs::exists(parent, ec_parent_exists) || ec_parent_exists) {
                    if (ec_parent_exists) logFilesystemError("createDirectory (parent exists check)", parent, ec_parent_exists);
                    else LOG_ERROR("Parent directory does not exist for non-recursive createDirectory: {}", parent.string());
                    return false;
                }
                std::error_code ec_parent_is_dir;
                if (!fs::is_directory(parent, ec_parent_is_dir) || ec_parent_is_dir) {
                     if (ec_parent_is_dir) logFilesystemError("createDirectory (parent is_directory check)", parent, ec_parent_is_dir);
                     else LOG_ERROR("Parent path is not a directory for non-recursive createDirectory: {}", parent.string());
                    return false;
                }
            }
            success = fs::create_directory(path, ec);
        }
        
        // If operation failed AND the error is not 'file_exists' (which we can treat as success for "ensuring" a directory)
        if (!success && ec && ec != std::errc::file_exists) {
            logFilesystemError(recursive ? "create_directories" : "create_directory", path, ec);
            return false;
        }
        // If it already existed or was successfully created, return true.
        // To be absolutely sure after a file_exists error, one might re-check it IS a directory.
        if (ec == std::errc::file_exists) {
            std::error_code ec_is_dir;
            if (!fs::is_directory(path, ec_is_dir) || ec_is_dir) {
                if (ec_is_dir) logFilesystemError("createDirectory (post-create file_exists type check)", path, ec_is_dir);
                else LOG_ERROR("Path {} exists but is not a directory after createDirectory reported file_exists.", path.string());
                return false; // It exists but it's not a directory, so creation effectively failed.
            }
        }
        return true;
    } catch (const fs::filesystem_error& e) {
         logFilesystemError(recursive ? "create_directories" : "create_directory", path, e.code());
        return false;
    }
}

bool FilesystemUtils::remove(const fs::path& path, bool recursive) {
    try {
        std::error_code ec_exists;
        if (!fs::exists(path, ec_exists)) { // Check if path exists first
            if (ec_exists) { // An error occurred trying to check existence
                logFilesystemError("remove (pre-check exists)", path, ec_exists);
                return false; // Don't proceed if existence check failed
            }
            // Path doesn't exist, original code treated this as success for remove. Maintain this.
            return true; 
        }
        // Path exists, proceed with removal.
        std::error_code ec_remove;
        bool success = false;
        if (recursive && FilesystemUtils::isDirectory(path)) { // Make sure isDirectory check is fine
            uintmax_t removed_count = fs::remove_all(path, ec_remove);
            success = !ec_remove; 
            if(success) LOG_INFO("Recursively removed {} entries from '{}'", removed_count, path.string());
        } else {
            success = fs::remove(path, ec_remove);
            if(success) LOG_INFO("Removed '{}'", path.string());
        }
        
        if (!success && ec_remove) { // No need to check for no_such_file_or_directory, already handled by pre-check
            logFilesystemError(recursive && FilesystemUtils::isDirectory(path) ? "remove_all" : "remove", path, ec_remove);
            return false; // Return false if removal failed for an existing file
        }
        return true; // Successfully removed or didn't exist
    } catch (const fs::filesystem_error& e) {
         logFilesystemError("remove operation", path, e.code());
        return false;
    }
}

bool FilesystemUtils::copyFile(const fs::path& source, const fs::path& destination, bool overwrite) {
    try {
        std::error_code ec_source_check;
        if (!FilesystemUtils::isFile(source)) { // Ensure source is a file
             if (fs::exists(source, ec_source_check) && !ec_source_check) // If it exists but not a file	
                 LOG_ERROR("Source for copyFile is not a regular file: {}", source.string());
             // else, isFile already logs if it doesn't exist or error in check
            return false;
        }

        auto destDir = destination.parent_path();
        if (!destDir.empty() && !FilesystemUtils::ensureDirectoryExists(destDir)) {
            LOG_ERROR("Failed to ensure destination directory exists for copyFile: {}", destDir.string());
            return false;
        }

        std::error_code ec_copy;
        fs::copy_options options = fs::copy_options::none;
        if (overwrite) {
            options |= fs::copy_options::overwrite_existing;
        } else {
            std::error_code ec_dest_exists;
            if (fs::exists(destination, ec_dest_exists) && !ec_dest_exists) {
                 LOG_ERROR("Destination file exists and overwrite is false for copyFile: {}", destination.string());
                 return false;
            } else if (ec_dest_exists) {
                logFilesystemError("copyFile (destination exists check)", destination, ec_dest_exists);
                return false;
            }
        }
        
        fs::copy_file(source, destination, options, ec_copy);
        if (ec_copy) {
            logFilesystemError("copy_file", source, ec_copy);
            return false;
        }
        LOG_INFO("Copied file '{}' to '{}'", source.string(), destination.string());
        return true;
    } catch (const fs::filesystem_error& e) {
        LOG_ERROR("Exception during copyFile from '{}' to '{}': {}", source.string(), destination.string(), e.what());
        return false;
    }
}

bool FilesystemUtils::copyDirectory(const fs::path& source, const fs::path& destination, bool recursive) {
    try {
        std::error_code ec_source_check;
        if (!fs::exists(source, ec_source_check) || ec_source_check || !fs::is_directory(source, ec_source_check) || ec_source_check) {
             if(ec_source_check) logFilesystemError("copyDirectory (source check)", source, ec_source_check);
             else LOG_ERROR("Source for copyDirectory is not a valid directory or does not exist: {}", source.string());
             return false;
        }

        // Ensure destination *base* directory exists (e.g. if copying dirA to new_parent/dirA, new_parent should exist)
        // If destination itself is the target directory name (e.g. copying dirS to dirD)
        // then dirD's parent must exist. If dirD exists, fs::copy handles it based on options.
        fs::path dest_parent_dir_to_ensure = destination.parent_path();
        if (fs::exists(destination) && fs::is_directory(destination)) {
            // If destination exists and is a directory, we're copying *into* it.
            // The parent is destination itself effectively, which exists.
        } else if (!dest_parent_dir_to_ensure.empty()) {
             if (!FilesystemUtils::ensureDirectoryExists(dest_parent_dir_to_ensure)) {
                LOG_ERROR("Failed to ensure destination parent directory exists for copyDirectory: {}", dest_parent_dir_to_ensure.string());
                return false;
            }
        }
        
        std::error_code ec_copy;
        fs::copy_options options = fs::copy_options::overwrite_existing; // Default from original example, can be changed
        if (recursive) {
            options |= fs::copy_options::recursive;
        }
        // To mimic common `cp -R src dest` behavior where if `dest` exists, `src` is copied inside it:
        // fs::copy(source, destination / source.filename(), options, ec_copy); // This is one interpretation
        // Or, if `dest` is the target name:
        fs::copy(source, destination, options, ec_copy);

        if (ec_copy) {
            logFilesystemError("copy directory", source, ec_copy);
            return false;
        }
        LOG_INFO("Copied directory '{}' to '{}' {}", source.string(), destination.string(), recursive ? "recursively" : "");
        return true;
    } catch (const fs::filesystem_error& e) {
        LOG_ERROR("Exception during copyDirectory from '{}' to '{}': {}", source.string(), destination.string(), e.what());
        return false;
    }
}

bool FilesystemUtils::moveFile(const fs::path& source, const fs::path& destination) {
    try {
        std::error_code ec_source_check;
        if (!fs::exists(source, ec_source_check) || ec_source_check) { // Check if source exists
             if(ec_source_check) logFilesystemError("moveFile (source exists check)", source, ec_source_check);
             else LOG_ERROR("Source path for moveFile does not exist: {}", source.string());
            return false;
        }
        // It's not strictly necessary to check if source is a file, as fs::rename can move directories too.
        // However, if the intent is *only* files, add: if (!FilesystemUtils::isFile(source)) return false;

        auto destDir = destination.parent_path();
        if (!destDir.empty() && !FilesystemUtils::ensureDirectoryExists(destDir)) {
            LOG_ERROR("Failed to ensure destination directory exists for moveFile: {}", destDir.string());
            return false;
        }

        std::error_code ec_rename;
        fs::rename(source, destination, ec_rename); 
        if (ec_rename) {
            // Original code had a check for destination exists if rename failed. This is a common cause.
            std::error_code ec_dest_exists;
            if (fs::exists(destination, ec_dest_exists) && !ec_dest_exists) {
                 LOG_ERROR("MoveFile failed: Destination '{}' already exists, and fs::rename does not typically overwrite across devices or if dest is a non-empty dir.", destination.string());
            } else {
                 logFilesystemError("moveFile (rename operation)", source, ec_rename);
            }
            return false;
        }
        LOG_INFO("Moved '{}' to '{}'", source.string(), destination.string());
        return true;
    } catch (const fs::filesystem_error& e) {
        LOG_ERROR("Exception during moveFile from '{}' to '{}': {}", source.string(), destination.string(), e.what());
        return false;
    }
}

// rename is essentially an alias for moveFile if we consider fs::rename behavior.
// If FilesystemUtils::rename should have different semantics (e.g. strictly no cross-device), it would need specific handling.
bool FilesystemUtils::rename(const fs::path& oldPath, const fs::path& newPath) {
    // This can be a direct call to moveFile if semantics are identical
    // return moveFile(oldPath, newPath);
    // Or, a direct fs::rename with its own error handling if preferred for distinct logging/behavior:
     try {
        std::error_code ec;
        fs::rename(oldPath, newPath, ec);
        if (ec) {
            logFilesystemError("rename", oldPath, ec);
            return false;
        }
        LOG_INFO("Renamed '{}' to '{}'", oldPath.string(), newPath.string());
        return true;
    } catch (const fs::filesystem_error& e) {
        LOG_ERROR("Exception during rename from '{}' to '{}': {}", oldPath.string(), newPath.string(), e.what());
        return false;
    }
}

bool FilesystemUtils::setLastModifiedTime(const fs::path& path, const fs::file_time_type& time) {
    try {
        std::error_code ec_exists;
        if (!fs::exists(path, ec_exists) || ec_exists) {
             if(ec_exists) logFilesystemError("setLastModifiedTime (exists check)", path, ec_exists);
             else LOG_ERROR("Cannot setLastModifiedTime, path does not exist: {}", path.string());
            return false;
        }
        std::error_code ec_set_time;
        fs::last_write_time(path, time, ec_set_time);
        if (ec_set_time) {
            logFilesystemError("set last write time", path, ec_set_time);
            return false;
        }
        return true;
    } catch (const fs::filesystem_error& e) {
        LOG_ERROR("Exception during setLastModifiedTime for path '{}': {}", path.string(), e.what());
        return false;
    }
}

bool FilesystemUtils::setPermissions(const fs::path& path, fs::perms permissions) {
    try {
        std::error_code ec_exists;
        if (!fs::exists(path, ec_exists) || ec_exists) {
             if(ec_exists) logFilesystemError("setPermissions (exists check)", path, ec_exists);
             else LOG_ERROR("Cannot setPermissions, path does not exist: {}", path.string());
            return false;
        }
        std::error_code ec_set_perms;
        fs::permissions(path, permissions, fs::perm_options::replace, ec_set_perms); // Use replace for clarity
        if (ec_set_perms) {
            logFilesystemError("setting permissions", path, ec_set_perms);
            return false;
        }
        return true;
    } catch (const fs::filesystem_error& e) {
        LOG_ERROR("Exception during setPermissions for path '{}': {}", path.string(), e.what());
        return false;
    }
}

std::string FilesystemUtils::createTempFile(const std::string& prefix, const std::string& extension) {
    try {
        std::string temp_dir_str = FilesystemUtils::getTempDirectory();
        if (temp_dir_str.empty()) {
            LOG_ERROR("Failed to get temporary directory for createTempFile.");
            return "";
        }
        fs::path temp_dir_path = temp_dir_str;

        // Construct pattern similar to original, but ensure it's just a filename pattern for OS functions
        std::string filename_pattern = prefix + "XXXXXX"; // OS functions usually handle the random part
        // The extension is handled *after* file creation by renaming, if needed and non-empty.

        #ifdef _WIN32
            char temp_filename_buf[MAX_PATH];
            // GetTempFileNameA creates a 0-byte file. We want the name.
            // It uses the provided path (temp_dir_str) and prefix.
            // The 0 means generate unique number. temp_filename_buf receives the name.
            if (GetTempFileNameA(temp_dir_str.c_str(), prefix.c_str(), 0, temp_filename_buf) != 0) {
                std::string created_filepath = temp_filename_buf;
                if (!extension.empty()) {
                    std::string target_filepath = created_filepath;
                    if (!extension.empty() && extension[0] != '.') {
                        target_filepath += ".";
                    }
                    target_filepath += extension;
                    // Rename the 0-byte file to include the extension.
                    // Using std::rename which maps to OS rename.
                    if (std::rename(created_filepath.c_str(), target_filepath.c_str()) == 0) {
                        return target_filepath;
                    } else {
                        LOG_ERROR("Failed to rename temporary file '{}' to '{}' to add extension. Errno: {}", created_filepath, target_filepath, errno);
                        fs::remove(created_filepath); // Cleanup original empty file
                        return "";
                    }
                }
                return created_filepath; // Return without extension if extension is empty
            } else {
                LOG_ERROR("Failed to create unique temporary file using GetTempFileNameA. OS Error: {}", GetLastError());
                return "";
            }
        #else // POSIX
            std::string template_str = (temp_dir_path / (filename_pattern + (extension.empty() ? "" : ("." + extension)))).string();
            // mkstemp requires a writable C string, and modifies it.
            // If an extension is desired from the start, it needs to be part of the XXXXXX template if mkstemp supports that, or rename after.
            // Simpler to use mkstemp for base, then rename if extension is fixed.
            // For more control: create path template prefix + XXXXXX, call mkstemp, then rename to add suffix.
            std::string mkstemp_template_str = (temp_dir_path / (prefix + "XXXXXX")).string();
            std::vector<char> template_cstr(mkstemp_template_str.begin(), mkstemp_template_str.end());
            template_cstr.push_back('\0');
            
            int fd = mkstemp(template_cstr.data());
            if (fd != -1) {
                close(fd); // We just need the filename, file is created empty.
                std::string created_filepath(template_cstr.data());
                if (!extension.empty()) {
                    std::string target_filepath = created_filepath;
                    if (extension[0] != '.') target_filepath += ".";
                    target_filepath += extension;
                    if (std::rename(created_filepath.c_str(), target_filepath.c_str()) == 0) {
                        return target_filepath;
                    } else {
                        LOG_ERROR("Failed to rename temporary file '{}' to '{}' to add extension. Errno: {}", created_filepath, target_filepath, errno);
                        fs::remove(created_filepath); // Cleanup
                        return "";
                    }
                }
                return created_filepath;
            } else {
                LOG_ERROR("Failed to create unique temporary file using mkstemp for template '{}'. Errno: {}", mkstemp_template_str, errno);
                return "";
            }
        #endif
    } catch (const fs::filesystem_error& e) {
        LOG_ERROR("Filesystem exception during createTempFile: {}", e.what());
        return "";
    } catch (const std::exception& e) {
        LOG_ERROR("General exception during createTempFile: {}", e.what());
        return "";
    }
}

std::string FilesystemUtils::createTempDirectory(const std::string& prefix) {
    try {
        std::string temp_dir_str = FilesystemUtils::getTempDirectory();
        if (temp_dir_str.empty()) {
            LOG_ERROR("Failed to get temporary directory for createTempDirectory.");
            return "";
        }
        fs::path temp_dir_path = temp_dir_str;
        std::string dirname_template_str = (temp_dir_path / (prefix + "XXXXXX")).string();

        #ifdef _WIN32
            // Windows: No direct mkdtemp. Create a unique name, then create dir.
            // This is less secure than mkdtemp which creates and sets permissions.
            // GetTempFileNameA can give a unique file name; we remove it and create a directory.
            char temp_placeholder_name[MAX_PATH];
            if (GetTempFileNameA(temp_dir_str.c_str(), prefix.c_str(), 0, temp_placeholder_name) != 0) {
                std::string potential_dir_path_str = temp_placeholder_name;
                fs::remove(potential_dir_path_str); // Remove the placeholder file created by GetTempFileNameA
                std::error_code ec_create_dir;
                if (fs::create_directory(potential_dir_path_str, ec_create_dir)) {
                    return potential_dir_path_str;
                } else {
                    logFilesystemError("create_directory for temp dir (Win32)", potential_dir_path_str, ec_create_dir);
                    return "";
                }
            } else {
                LOG_ERROR("Failed to generate unique name for temporary directory using GetTempFileNameA. OS Error: {}", GetLastError());
                return "";
            }
        #else // POSIX
            std::vector<char> template_cstr(dirname_template_str.begin(), dirname_template_str.end());
            template_cstr.push_back('\0');
            if (mkdtemp(template_cstr.data()) != nullptr) {
                return std::string(template_cstr.data());
            } else {
                LOG_ERROR("Failed to create unique temporary directory using mkdtemp for template '{}'. Errno: {}", dirname_template_str, errno);
                return "";
            }
        #endif
    } catch (const fs::filesystem_error& e) {
        LOG_ERROR("Filesystem exception during createTempDirectory: {}", e.what());
        return "";
    } catch (const std::exception& e) {
        LOG_ERROR("General exception during createTempDirectory: {}", e.what());
        return "";
    }
}

} // namespace common_utils
} // namespace oscean 