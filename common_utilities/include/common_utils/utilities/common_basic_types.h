#pragma once

/**
 * @file common_basic_types.h
 * @brief Defines truly universal basic data types, enums, or auxiliary structs for common_utilities.
 *
 * This file is intended for NEW, PURELY GENERAL types identified during the refactoring of
 * the common_utilities module that DO NOT exist in or are not suitable to be directly used from
 * core_service_interfaces/include/core_services/common_data_types.h.
 *
 * If common_utilities needs types already defined in core_services/common_data_types.h,
 * it will directly #include <core_services/common_data_types.h> (or the project-specific
 * path to that file).
 *
 * This file might remain empty or contain very few definitions if no such new, general-purpose
 * types are identified as necessary for common_utilities internal workings.
 */

// Example of a new, purely general type that might be needed by common_utilities
// and is not domain-specific or present in core_services/common_data_types.h:
/*
namespace oscean_common
{
namespace utils
{

    enum class GeneralStatusCode {
        SUCCESS = 0,
        UNKNOWN_ERROR = 1,
        NOT_IMPLEMENTED = 2,
        INVALID_ARGUMENT = 3,
        TIMEOUT = 4,
        RESOURCE_UNAVAILABLE = 5
        // Add other general status codes as needed
    };

    struct VersionInfo {
        int major = 0;
        int minor = 0;
        int patch = 0;
        std::string build_metadata;
    };

} // namespace utils
} // namespace oscean_common
*/

// Add new, purely general types below as identified during refactoring. 