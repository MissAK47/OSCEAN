#include "common_utils/memory/memory_statistics.h"
#include <sstream>
#include <iomanip>

namespace oscean::common_utils::memory {

// === MemoryStatisticsCollector实现 ===

void MemoryStatisticsCollector::recordAllocation(size_t size, const std::string& tag) {
    std::lock_guard<std::mutex> lock(statsMutex_);
    
    // 使用简化的统计字段
    overallStats_.allocationCount++;
    overallStats_.currentUsed += size;
    overallStats_.totalAllocated += size;
    
    if (overallStats_.currentUsed > overallStats_.peakUsage) {
        overallStats_.peakUsage = overallStats_.currentUsed;
    }
}

void MemoryStatisticsCollector::recordDeallocation(size_t size, const std::string& tag) {
    std::lock_guard<std::mutex> lock(statsMutex_);
    
    // 使用简化的统计字段
    overallStats_.deallocationCount++;
    if (size > 0 && overallStats_.currentUsed >= size) {
        overallStats_.currentUsed -= size;
    }
}

MemoryUsageStats MemoryStatisticsCollector::getOverallStats() const {
    std::lock_guard<std::mutex> lock(statsMutex_);
    return overallStats_;
}

void MemoryStatisticsCollector::reset() {
    std::lock_guard<std::mutex> lock(statsMutex_);
    overallStats_ = MemoryUsageStats{};
}

std::string MemoryStatisticsCollector::generateReport() const {
    std::lock_guard<std::mutex> lock(statsMutex_);
    
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    
    oss << "=== Memory Statistics Report ===" << std::endl;
    oss << "Total Allocated: " << overallStats_.totalAllocated << " bytes" << std::endl;
    oss << "Currently Used: " << overallStats_.currentUsed << " bytes" << std::endl;
    oss << "Peak Usage: " << overallStats_.peakUsage << " bytes" << std::endl;
    oss << "Allocation Count: " << overallStats_.allocationCount << std::endl;
    oss << "Deallocation Count: " << overallStats_.deallocationCount << std::endl;
    oss << "Fragmentation Ratio: " << overallStats_.fragmentationRatio * 100 << "%" << std::endl;
    
    return oss.str();
}

} // namespace oscean::common_utils::memory 