// @file  nncov_pool_blas.hpp
// @brief MPN-COV 
// @author Jiangtao Xie
// @author Peihua Li

/*
Copyright (C) 2017 Peihua Li and Jiangtao Xie

All rights reserved.
*/

#ifndef __vl__nncov__pool__blas__
#define __vl__nncov__pool__blas__

#include "../data.hpp"
#include <cstddef>

namespace vl { namespace impl {
    template<vl::DeviceType dev ,typename type,vl::DataType dataType>
    struct cov_pool
    {
        static vl::ErrorCode
            forward(Context& context,
                    type* output,
                    type const* data,
                    size_t height, size_t width, size_t depth, size_t num);
        static vl::ErrorCode
            backward(Context& context,
                     type* derData,
                     type const* data,
                     type const* derOutput,
                     size_t height, size_t width, size_t depth, size_t num);
    };
} }


#endif /* __vl_mpn_cov__ */
