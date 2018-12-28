// @file  nnmpn_cov_blas.hpp
// @brief MPN-COV 
// @author Jiangtao Xie
// @author Peihua Li

/*
Copyright (C) 2017 Peihua Li and Jiangtao Xie

All rights reserved.
*/

#ifndef __vl__nncov__traceNorm__blas__
#define __vl__nncov__traceNorm__blas__

#include "../data.hpp"
#include <cstddef>

namespace vl { namespace impl {
    template<vl::DeviceType dev ,typename type,vl::DataType dataType>
    struct cov_traceNorm
    {
        static vl::ErrorCode
            forward(Context& context,
                    type* output,
                    type const* data,
					type* aux_T,
                    size_t height, size_t width, size_t depth, size_t num);
        static vl::ErrorCode
            backward(Context& context,
                     type* derData,
                     type const* data,
                     type const* derOutput,
					 type const* derOutput_aux,
					 type const* aux_T,
                     size_t height, size_t width, size_t depth, size_t num);
		static vl::ErrorCode
			forward_aux(Context& context,
			            type* output,
						type const* data,
						type* aux_T,
						size_t height, size_t width, size_t depth, size_t num);
		static vl::ErrorCode
			backward_aux(Context& context,
			             type* derData,
						 type* derData_aux,
						 type const* data,
						 type const* derOutput,
						 type const* aux_T,
						 size_t height, size_t width, size_t depth, size_t num);
    };
} }


#endif /* __vl_mpn_cov__ */
