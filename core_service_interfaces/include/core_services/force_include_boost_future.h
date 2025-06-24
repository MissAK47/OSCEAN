// force_include_boost_future.h
#ifndef FORCE_INCLUDE_BOOST_FUTURE_H
#define FORCE_INCLUDE_BOOST_FUTURE_H

// 尝试解除任何可能的 future 宏定义
#ifdef future
#undef future
#endif

// 包含 boost::future
#include <boost/thread/future.hpp>

#endif // FORCE_INCLUDE_BOOST_FUTURE_H 