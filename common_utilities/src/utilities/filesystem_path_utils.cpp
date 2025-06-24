#include "common_utils/utilities/filesystem_utils.h"
#include "common_utils/utilities/logging_utils.h" // For LOG_ERROR, LOG_WARN if used by these functions
#include <filesystem>
#include <string>
#include <system_error> // For std::error_code in some fs operations

// Note: logFilesystemError is not directly used by these functions, so it's omitted here.
// It will be included in other split files where needed (e.g., io_utils, manipulation_utils)
// or defined as a static helper there.

namespace oscean {
namespace common_utils {

namespace fs = std::filesystem;

// Helper function to log filesystem errors - moved to relevant cpp files or a shared internal header if many use it.
// For now, assuming it's not directly called by the functions below.
// If any of these path functions *do* need a simplified error logging that doesn't fit LOG_ERROR,
// it would be a local static helper.

std::string FilesystemUtils::getFileExtension(const fs::path& path) {
    std::string ext = path.extension().string();
    if (!ext.empty() && ext[0] == '.') {
        return ext.substr(1); // Remove the leading dot
    }
    return ext; // Return as is if empty or doesn't start with a dot
}

std::string FilesystemUtils::getFileName(const fs::path& path, bool withExtension) {
    if (withExtension) {
        return path.filename().string();
    } else {
        return path.stem().string();
    }
}

std::string FilesystemUtils::getFileNameWithoutExtension(const fs::path& path) {
    return path.stem().string();
}

std::string FilesystemUtils::getParentDirectory(const fs::path& path) {
    return path.parent_path().string();
}

std::string FilesystemUtils::getDirectoryPath(const fs::path& path) {
    // This function relies on isDirectory. Assuming isDirectory is still part of FilesystemUtils
    // and its declaration is in filesystem_utils.h, to be implemented in filesystem_query_utils.cpp
    if (FilesystemUtils::isDirectory(path)) { // Corrected to call the static member
        return path.string();
    } else {
        return path.parent_path().string();
    }
}

std::string FilesystemUtils::combinePath(const fs::path& path1, const fs::path& path2) {
    return (path1 / path2).make_preferred().lexically_normal().string();
}

std::string FilesystemUtils::normalizePath(const fs::path& path) {
    std::error_code ec;
    fs::path weakly_canonical_path = fs::weakly_canonical(path, ec);
    if (ec) {
        // According to cppreference, weakly_canonical can throw or set ec.
        // If it sets ec and returns an empty path, or if it throws, we might want to log.
        // For now, returning original path's string representation on error, consistent with original.
        // logFilesystemError("normalizing path (weakly_canonical)", path, ec); // If logFilesystemError were here
        LOG_WARN("Filesystem error during normalizePath (weakly_canonical) on path '{}': {} ({})",
                 path.string(), ec.message(), ec.value());
        return path.string(); // Fallback or rethrow, original returned path.string() on exception
    }
    return weakly_canonical_path.string();
}

std::string FilesystemUtils::getRelativePath(const fs::path& path, const fs::path& basePath) {
    try {
        fs::path relative_path = fs::relative(path, basePath);
        return relative_path.string();
    } catch (const fs::filesystem_error& e) {
        // Original code logged and returned "", let's be consistent
        LOG_ERROR("Filesystem error during getRelativePath for '{}' relative to '{}': {}",
                  path.string(), basePath.string(), e.what());
        return ""; 
    }
}

std::string FilesystemUtils::getAbsolutePath(const fs::path& path) {
    try {
        std::error_code ec;
        fs::path absolute_path = fs::absolute(path, ec);
        if(ec) {
            // logFilesystemError("getting absolute path", path, ec); // If logFilesystemError were here
            LOG_ERROR("Filesystem error during getAbsolutePath on path '{}': {} ({})", 
                      path.string(), ec.message(), ec.value());
            return path.string(); // Return original on error, consistent with original
        }
        return absolute_path.string();
    } catch (const fs::filesystem_error& e) { // Catch specific filesystem_error
         LOG_ERROR("Exception during getAbsolutePath for path '{}': {}", path.string(), e.what());
         return path.string(); // Return original on exception, consistent with original
    }
}

std::string FilesystemUtils::getCurrentWorkingDirectory() {
    try {
        std::error_code ec;
        fs::path cwd = fs::current_path(ec);
        if(ec) {
            LOG_ERROR("Failed to get current working directory: {} ({})", ec.message(), ec.value());
            return "";
        }
        return cwd.string();
    } catch (const fs::filesystem_error& e) { // Catch specific filesystem_error
        LOG_ERROR("Exception during getCurrentWorkingDirectory: {}", e.what());
        return "";
    }
}

bool FilesystemUtils::setCurrentWorkingDirectory(const fs::path& path) {
    try {
        std::error_code ec;
        fs::current_path(path, ec);
        if(ec) {
            // logFilesystemError("setting current working directory", path, ec); // If logFilesystemError were here
            LOG_ERROR("Filesystem error during setCurrentWorkingDirectory to '{}': {} ({})", 
                      path.string(), ec.message(), ec.value());
            return false;
        }
        return true;
    } catch (const fs::filesystem_error& e) { // Catch specific filesystem_error
        LOG_ERROR("Exception during setCurrentWorkingDirectory to '{}': {}", path.string(), e.what());
        return false;
    }
}

std::string FilesystemUtils::getTempDirectory() {
    try {
        std::error_code ec;
        fs::path temp_dir = fs::temp_directory_path(ec);
        if(ec) {
            LOG_ERROR("Failed to get temporary directory path: {} ({})", ec.message(), ec.value());
            return "";
        }
        return temp_dir.string();
    } catch (const fs::filesystem_error& e) { // Catch specific filesystem_error
        LOG_ERROR("Exception during getTempDirectory: {}", e.what());
        return "";
    }
}

} // namespace common_utils
} // namespace oscean 