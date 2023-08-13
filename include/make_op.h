//
// Created by 赵丹 on 2023/8/3.
//

#ifndef TVM_TEST_MAKE_OP_H
#define TVM_TEST_MAKE_OP_H

#include "tvm/relay/attrs/nn.h"
#include "tvm/relay/op.h"

#include <string>
#include <utility>
#include <vector>

namespace tvm::relay {
    template<typename T>
    inline Expr MakeConv(Expr data, Expr weight, const Array<IndexExpr> &strides, const Array<IndexExpr> &padding,
                         const Array<IndexExpr> &dilation, int groups, const IndexExpr &channels,
                         const Array<IndexExpr> &kernel_size, const std::string &data_layout,
                         const std::string &kernel_layout, const std::string &out_layout, const DataType &out_dtype,
                         std::string op_name) {
        auto attrs = make_object<T>();
        attrs->strides = strides;
        attrs->padding = padding;
        attrs->dilation = dilation;
        attrs->groups = groups;
        attrs->channels = channels;
        attrs->kernel_size = kernel_size;
        attrs->data_layout = data_layout;
        attrs->kernel_layout = kernel_layout;
        attrs->out_layout = out_layout;
        attrs->out_dtype = out_dtype;
        const Op &op = Op::Get(std::move(op_name));
        return Call(op, {std::move(data), std::move(weight)}, Attrs(attrs), {});
    }

    inline Expr MakeAdd(const relay::Expr &lhs, const relay::Expr &rhs) {
        const Op &add_op = Op::Get("add");
        return Call(add_op, {lhs, rhs});
    }
}

#endif //TVM_TEST_MAKE_OP_H
