#include "common_utils/utilities/filesystem_utils.h"
#include "common_utils/utilities/logging_utils.h"
#include <filesystem>
#include <fstream>
#include <sstream> // For readFileToString
#include <vector>
#include <system_error> // For std::error_code

namespace oscean {
namespace common_utils {

namespace fs = std::filesystem;

// Static helper function, similar to the one in filesystem_query_utils.cpp
// If this becomes common across many files, consider a shared internal utility header.
static void logFilesystemError(const std::string& operation, const fs::path& path, const std::error_code& ec) {
    std::string path_str = path.string();
    LOG_ERROR("Filesystem error during '{}' on path '{}': {} ({})", 
              operation, path_str, ec.message(), ec.value());
}

// ensureDirectoryExists needs to be defined before it's used by write*ToFile functions
// It relies on isDirectory and createDirectories (which will be in manipulation_utils)
// For now, we assume FilesystemUtils::isDirectory is available (declared in .h, implemented in query_utils)
// and FilesystemUtils::createDirectory (recursive) is available (declared in .h, to be impl in manipulation_utils)

bool FilesystemUtils::ensureDirectoryExists(const fs::path& directory) {
    if (directory.empty()) {
        LOG_ERROR("Attempted to ensure an empty directory path exists.");
        return false;
    }
    
    std::error_code ec_exists;
    if (fs::exists(directory, ec_exists)) {
        if (ec_exists) {
            logFilesystemError("ensureDirectoryExists (exists check)", directory, ec_exists);
            return false; // Error checking existence
        }
        std::error_code ec_is_dir;
        if (fs::is_directory(directory, ec_is_dir)) {
            if (ec_is_dir) {
                 logFilesystemError("ensureDirectoryExists (is_directory check)", directory, ec_is_dir);
                 return false; // Error checking if it's a directory
            }
            return true; // Exists and is a directory
        }
        // Exists but is not a directory
        LOG_ERROR("Path exists but is not a directory: {}", directory.string());
        return false;
    }
    
    // Path does not exist, try to create it.
    // This part implicitly calls createDirectory (recursive = true), which will be in manipulation_utils.
    // To avoid a direct call to a function in another cpp before it's defined in this split sequence,
    // we replicate the core logic of create_directories here for now.
    // A better long-term solution might be to ensure manipulation_utils is processed first if there are strict dependencies,
    // or to have ensureDirectoryExists call FilesystemUtils::createDirectory(directory, true).
    // For this step, we'll use fs::create_directories directly.
    std::error_code ec_create;
    if (fs::create_directories(directory, ec_create)) {
        LOG_INFO("Created directory: {}", directory.string());
        return true;
    } else {
        // If create_directories failed, but it was because it already exists (race condition?)
        // then re-check existence and type.
        if (ec_create == std::errc::file_exists) { 
            std::error_code ec_is_dir_after_create_fail;
            if(fs::is_directory(directory, ec_is_dir_after_create_fail) && !ec_is_dir_after_create_fail) return true;
        }
        logFilesystemError("create_directories in ensureDirectoryExists", directory, ec_create);
        return false;
    }
}

bool FilesystemUtils::writeStringToFile(const fs::path& filePath, const std::string& content, bool append) {
    try {
        auto parentDir = filePath.parent_path();
        if (!parentDir.empty() && !ensureDirectoryExists(parentDir)) {
             LOG_ERROR("Failed to ensure parent directory exists for file: {}", filePath.string());
             return false;
        }
        
        std::ios_base::openmode mode = std::ios::out;
        if (append) {
            mode |= std::ios::app;
        }
        
        std::ofstream file(filePath, mode);
        if (!file.is_open()) {
            LOG_ERROR("Failed to open file for writing: {}", filePath.string());
            return false;
        }
        
        file << content;
        bool success = file.good(); // Check stream state after write
        file.close(); 
        if (!success || file.fail()) { // Check failbit too after close
            LOG_ERROR("Failed to write content completely to file: {}", filePath.string());
            return false; // Explicitly return false on failure
        }
        return true;
    }
    catch (const std::ios_base::failure& e) { // Catch specific stream exceptions
         LOG_ERROR("IOS Exception during writeStringToFile for path '{}': {} ({})", filePath.string(), e.what(), e.code().message());
        return false;
    }
}

bool FilesystemUtils::writeBinaryToFile(const fs::path& path, const std::vector<char>& data, bool append) {
    try {
        auto parentDir = path.parent_path();
        if (!parentDir.empty() && !ensureDirectoryExists(parentDir)) {
             LOG_ERROR("Failed to ensure parent directory exists for file: {}", path.string());
             return false;
        }
        
        std::ios_base::openmode mode = std::ios::binary | std::ios::out;
        if (append) {
            mode |= std::ios::app;
        }
        
        std::ofstream file(path, mode);
        if (!file.is_open()) {
             LOG_ERROR("Failed to open file for binary writing: {}", path.string());
            return false;
        }
        
        if (!data.empty()) {
            file.write(data.data(), data.size());
        }
        
        bool success = file.good();
        file.close();
        if (!success || file.fail()) {
            LOG_ERROR("Failed to write binary content completely to file: {}", path.string());
            return false;
        }
        return true;
    }
    catch (const std::ios_base::failure& e) {
         LOG_ERROR("IOS Exception during writeBinaryToFile for path '{}': {} ({})", path.string(), e.what(), e.code().message());
        return false;
    }
}

std::optional<std::string> FilesystemUtils::readFileToString(const fs::path& filePath) {
    std::error_code ec_check;
    if (!fs::exists(filePath, ec_check) || ec_check || !fs::is_regular_file(filePath, ec_check) || ec_check) {
        if (ec_check) logFilesystemError("readFileToString (pre-check)", filePath, ec_check);
        else LOG_ERROR("File not found or is not a regular file for readFileToString: {}", filePath.string());
        return std::nullopt;
    }

    std::ifstream fileStream(filePath);
    if (!fileStream.is_open()) {
        LOG_ERROR("Failed to open file for readFileToString: {}", filePath.string());
        return std::nullopt;
    }

    std::stringstream buffer;
    try {
        buffer << fileStream.rdbuf();
        if (fileStream.bad()) { // Check for badbit after reading
            LOG_ERROR("Stream badbit set after reading file '{}' to string.", filePath.string());
            return std::nullopt;
        }
        return buffer.str();
    } catch (const std::ios_base::failure& e) {
        LOG_ERROR("IOS failure reading file '{}' to string: {} ({})", filePath.string(), e.what(), e.code().message());
        return std::nullopt;
    }
}

std::optional<std::vector<unsigned char>> FilesystemUtils::readFileToBinary(const fs::path& filePath) {
    std::error_code ec_check;
    if (!fs::exists(filePath, ec_check) || ec_check || !fs::is_regular_file(filePath, ec_check) || ec_check) {
        if (ec_check) logFilesystemError("readFileToBinary (pre-check)", filePath, ec_check);
        else LOG_ERROR("File not found or is not a regular file for readFileToBinary: {}", filePath.string());
        return std::nullopt;
    }

    std::ifstream fileStream(filePath, std::ios::binary | std::ios::ate);
    if (!fileStream.is_open()) {
        LOG_ERROR("Failed to open file in binary mode for readFileToBinary: {}", filePath.string());
        return std::nullopt;
    }

    std::streamsize size = fileStream.tellg();
    if (size < 0) { 
        LOG_ERROR("Failed to determine file size (tellg returned negative) for: {}", filePath.string());
        return std::nullopt;
    }
    if (size == 0) { // Handle empty file explicitly
        return std::vector<unsigned char>();
    }
    
    fileStream.seekg(0, std::ios::beg);

    std::vector<unsigned char> buffer(static_cast<size_t>(size));
    try {
        fileStream.read(reinterpret_cast<char*>(buffer.data()), size);
        // Check if the correct number of bytes was read
        if (fileStream.gcount() != size) {
            LOG_ERROR("Failed to read the full binary content (gcount {} != size {}) from file: {}", 
                      fileStream.gcount(), size, filePath.string());
            return std::nullopt;
        }
        if (fileStream.bad()) {
            LOG_ERROR("Stream badbit set after reading binary file '{}'.", filePath.string());
            return std::nullopt;
        }
        return buffer;
    } catch (const std::ios_base::failure& e) {
        LOG_ERROR("IOS failure reading binary file '{}': {} ({})", filePath.string(), e.what(), e.code().message());
        return std::nullopt;
    }
}

} // namespace common_utils
} // namespace oscean 