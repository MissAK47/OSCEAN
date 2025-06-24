#pragma once

// Include necessary standard library headers for hash specializations
#include <string>
#include <vector>
#include <optional>
#include <cstddef> // For std::size_t
#include <functional> // For std::hash
#include <chrono> // For TimeRange 的 time_point 支持

// Forward declarations to avoid circular includes
namespace oscean::core_services {
    struct CRSInfo;
    struct BoundingBox;
    struct IndexRange;
    struct TimeRange;
    struct DataChunkKey;
    using Timestamp = uint64_t; // Define Timestamp type here to match common_data_types.h
}

// It's crucial that this file is included *after* all standard library headers
// that define the primary templates for std::hash (like <string>, <vector>, etc.)
// and *after* the oscean::core_services types are fully defined.

namespace oscean {
namespace core_services {

// Hash Combine Utility (Boost Style)
// Placed here to be available for all specializations in this file.
template <class T>
inline void hash_combine(std::size_t& seed, const T& v) {
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

} // namespace core_services
} // namespace oscean

namespace std {

    /**
     * @brief Hash specialization for CRSInfo.
     */
    template <>
    struct hash<oscean::core_services::CRSInfo> {
        inline std::size_t operator()(const oscean::core_services::CRSInfo& crs) const noexcept {
            // Simplified for now, ensure all relevant fields for equality are included
            std::size_t seed = 0;
            oscean::core_services::hash_combine(seed, crs.id);
            oscean::core_services::hash_combine(seed, crs.name); // Example: add more fields
            return seed;
        }
    };

    /**
     * @brief Hash specialization for BoundingBox.
     */
    template <>
    struct hash<oscean::core_services::BoundingBox> {
        inline std::size_t operator()(const oscean::core_services::BoundingBox& bbox) const noexcept {
            std::size_t seed = 0;
            oscean::core_services::hash_combine(seed, bbox.minX);
            oscean::core_services::hash_combine(seed, bbox.minY);
            oscean::core_services::hash_combine(seed, bbox.maxX);
            oscean::core_services::hash_combine(seed, bbox.maxY);
            if (bbox.minZ.has_value()) { // Use .has_value() for optional
                oscean::core_services::hash_combine(seed, bbox.minZ.value());
            }
            if (bbox.maxZ.has_value()) { // Use .has_value() for optional
                oscean::core_services::hash_combine(seed, bbox.maxZ.value());
            }
            oscean::core_services::hash_combine(seed, bbox.crsId);
            return seed;
        }
    };

    /**
     * @brief Hash specialization for IndexRange.
     */
    template <>
    struct hash<oscean::core_services::IndexRange> {
        inline std::size_t operator()(const oscean::core_services::IndexRange& range) const noexcept {
            std::size_t seed = 0;
            oscean::core_services::hash_combine(seed, range.start);
            oscean::core_services::hash_combine(seed, range.count);
            return seed;
        }
    };

    /**
     * @brief Hash specialization for TimeRange. 
     */
    template <>
    struct hash<oscean::core_services::TimeRange> {
         inline std::size_t operator()(const oscean::core_services::TimeRange& range) const noexcept {
            std::size_t seed = 0;
            // TimeRange 现在使用 startTime 和 endTime (chrono::system_clock::time_point)
            auto startCount = range.startTime.time_since_epoch().count();
            auto endCount = range.endTime.time_since_epoch().count();
            oscean::core_services::hash_combine(seed, startCount); 
            oscean::core_services::hash_combine(seed, endCount);
            return seed;
        }
    };
    
    // Placeholder for std::hash<oscean::core_services::Timestamp> if not already provided by chrono or elsewhere
    // If Timestamp is, for example, std::chrono::system_clock::time_point, this might not be needed
    // or needs to be adapted based on Timestamp's actual definition.
    // template <>
    // struct hash<oscean::core_services::Timestamp> {
    //     inline std::size_t operator()(const oscean::core_services::Timestamp& ts) const noexcept {
    //         return std::hash<long long>()(ts.time_since_epoch().count()); // Example if it's a time_point
    //     }
    // };

    /**
     * @brief Hash specialization for DataChunkKey.
     */
    template <>
    struct hash<oscean::core_services::DataChunkKey> { 
        inline std::size_t operator()(const oscean::core_services::DataChunkKey& k) const noexcept {
            std::size_t seed = 0;
            oscean::core_services::hash_combine(seed, k.filePath);
            oscean::core_services::hash_combine(seed, k.variableName);
            
            if (k.timeIndexRange.has_value()) {
                oscean::core_services::hash_combine(seed, k.timeIndexRange.value());
            }
            if (k.boundingBox.has_value()) {
                oscean::core_services::hash_combine(seed, k.boundingBox.value());
            }
            if (k.levelIndexRange.has_value()) {
                oscean::core_services::hash_combine(seed, k.levelIndexRange.value());
            }
            oscean::core_services::hash_combine(seed, k.requestDataType);
            return seed;
        }
    };

} // namespace std 