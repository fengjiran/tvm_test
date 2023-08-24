//
// Created by 赵丹 on 2023/8/3.
//

#ifndef TVM_TEST_MAKE_OP_H
#define TVM_TEST_MAKE_OP_H

#include "tvm/relay/attrs/nn.h"
#include "tvm/relay/attrs/transform.h"
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

    inline Expr MakeAdd(const Expr &lhs, const Expr &rhs) {
        const Op &add_op = Op::Get("add");
        return Call(add_op, {lhs, rhs});
    }

    inline Expr MakeRelu(const Expr &data) {
        const Op &op = Op::Get("nn.relu");
        return Call(op, {data}, Attrs(), {});
    }

    inline Expr MakeBatchNorm(Expr data, Expr gamma, Expr beta, Expr moving_mean, Expr moving_var, int axis,
                              double epsilon, bool center, bool scale) {
        auto attrs = make_object<BatchNormAttrs>();
        attrs->axis = axis;
        attrs->epsilon = epsilon;
        attrs->center = center;
        attrs->scale = scale;
        static const Op &op = Op::Get("nn.batch_norm");
        return Call(op,
                    {std::move(data), std::move(gamma), std::move(beta), std::move(moving_mean), std::move(moving_var)},
                    Attrs(attrs), {});
    }

    inline Expr MakeMultiply(Expr lhs, Expr rhs) {
        static const Op &op = Op::Get("multiply");
        return Call(op, {std::move(lhs), std::move(rhs)}, Attrs(), {});
    }

    inline Expr MakeConcatenate(Expr data, int axis) {
        auto attrs = make_object<ConcatenateAttrs>();
        attrs->axis = axis;
        static const Op &op = Op::Get("concatenate");
        return Call(op, {std::move(data)}, Attrs(attrs), {});
    }
}

#endif //TVM_TEST_MAKE_OP_H
