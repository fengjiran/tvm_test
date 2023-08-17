//
// Created by richard on 8/2/23.
//
#include "gtest/gtest.h"
#include "tvm/relay/expr.h"
#include "tvm/relay/op.h"
#include "tvm/relay/attrs/on_device.h"
#include "tvm/relay/op_attr_types.h"
#include "tvm/relay/op_strategy.h"
#include "tvm/target/virtual_device.h"
#include "tvm/topi/broadcast.h"
#include "tvm/topi/generic/injective.h"
#include "build_relay_model.h"

using namespace tvm;
using namespace tvm::relay;

/*! \brief Result of \p GetOnDeviceProps. */
struct OnDeviceProps {
    Expr body;  // = null
    VirtualDevice virtual_device = VirtualDevice::FullyUnconstrained();
    bool constrain_result = false;
    bool constrain_body = false;

    OnDeviceProps() = default;

    OnDeviceProps(Expr body, VirtualDevice virtual_device, bool constrain_result, bool constrain_body)
            : body(std::move(body)),
              virtual_device(std::move(virtual_device)),
              constrain_result(constrain_result),
              constrain_body(constrain_body) {}

    bool is_fixed() const { return constrain_result && constrain_body; }

    bool is_normal() const { return !constrain_result && constrain_body; }
};

const Op &OnDeviceOp() {
    static const Op &op = Op::Get("on_device");
    return op;
}

OnDeviceProps GetOnDeviceProps(const CallNode *call_node) {
    if (call_node->op == OnDeviceOp()) {
        ICHECK_EQ(call_node->args.size(), 1) << "on_device expects one argument";
        ICHECK(call_node->attrs.defined()) << "on_device requires attributes";
        const auto *on_device_attrs = call_node->attrs.as<OnDeviceAttrs>();
        ICHECK(on_device_attrs != nullptr) << "on_device requires OnDeviceAttrs";
        return {call_node->args[0], on_device_attrs->virtual_device, on_device_attrs->constrain_result,
                on_device_attrs->constrain_body};
    }
    return {};
}

OnDeviceProps GetOnDeviceProps(const Expr &expr) {
    if (const auto *call_node = expr.as<CallNode>()) {
        return GetOnDeviceProps(call_node);
    }
    return {};
}

/*!
 * \brief Returns \p expr as \p NodeType, or null if it is not of that type. Looks through
 * any "on_device" annotations.
 */
template<typename NodeType>
const NodeType *AsIgnoringOnDevice(const Expr &expr) {
    const auto *node = expr.as<NodeType>();
    if (node != nullptr) {
        return node;
    }
    OnDeviceProps props = GetOnDeviceProps(expr);
    if (!props.body.defined()) {
        return nullptr;
    }
    return props.body.as<NodeType>();
}

/*!
 * \brief Returns whether \p expr is a literal \p Constant, optionally wrapped by an "on_device"
 * annotation CallNode (which serves only to associate an \p VirtualDevice to the constant and has
 * no operational effect).
 */
bool IsSimpleConstant(const Expr &expr) {
    return AsIgnoringOnDevice<ConstantNode>(expr) != nullptr;
}

/*!
 * \brief Returns whether \p expr \p IsSimpleConstant directly or is a tuple of
 * \p IsComplexConstant expressions.
 */
bool IsComplexConstant(const Expr &expr) {
    if (IsSimpleConstant(expr)) {
        return true;
    } else if (const auto *tuple_node = AsIgnoringOnDevice<TupleNode>(expr)) {
        return std::all_of(tuple_node->fields.begin(), tuple_node->fields.end(), IsComplexConstant);
    } else {
        return false;
    }
}

TVM_REGISTER_GLOBAL("test.add_strategy")
.set_body_typed([](const Attrs& attrs, const Array<te::Tensor>& inputs, const Type& out_type,
                   const Target& target) {
    FTVMCompute fcompute = [](const Attrs& attrs, const Array<te::Tensor>& inputs,
                              const Type& out_type) -> Array<te::Tensor> {
        ICHECK_EQ(inputs.size(), 2U);
        return {topi::add(inputs[0], inputs[1])};
    };
    FTVMSchedule fschedule = [](const Attrs& attrs, const Array<te::Tensor>& outs,
                                const Target& target) {
        With<Target> target_scope(target);
        return topi::generic::schedule_injective(target, outs);
    };

    auto n = make_object<OpStrategyNode>();
    auto strategy = relay::OpStrategy(std::move(n));
    strategy.AddImplementation(fcompute, fschedule, "test.add_strategy", 10);
    return strategy;
});

TVM_REGISTER_GLOBAL("relay.backend.lower_call")
.set_body_typed([](const relay::Call& call, const Array<te::Tensor>& inputs,
                   const Target& target) {
    static auto fstrategy = Op::GetAttrMap<relay::FTVMStrategy>("FTVMStrategy");
    Op op = Downcast<Op>(call->op);
    auto out_type = call->checked_type();
    OpStrategy strategy = fstrategy[op](call->attrs, inputs, out_type, target);
    auto impl = strategy->specializations[0]->implementations[0];
    auto outs = impl.Compute(call->attrs, inputs, out_type);
    auto f = runtime::Registry::Get("relay.backend._make_LoweredOutput");
    if (!f) {
        LOG(FATAL) << "relay.backend._make_LoweredOutput is not registered";
    }
    return (*f)(outs, impl);
});

TEST(RelayPass, ConstantCheck) {
    relay::Constant c1 = relay::Constant(runtime::NDArray::Empty({1, 16, 64, 64},
                                                                 {kDLFloat, 32, 1},
                                                                 {kDLCPU, 0}));
    relay::Var x1 = relay::Var("x1",
                               TensorType({1, 16, 64, 64},
                                          DataType::Float(32)));
    ICHECK(IsComplexConstant(c1));
    ICHECK(!IsComplexConstant(x1));
}

TEST(RelayPass, FoldConstant) {
    auto reg_op_attr = runtime::Registry::Get("ir.RegisterOpAttr");
    ICHECK_NOTNULL(reg_op_attr);
    auto reset_op_attr = runtime::Registry::Get("ir.OpResetAttr");
    ICHECK_NOTNULL(reset_op_attr);
    auto add_op_strategy = runtime::Registry::Get("test.add_strategy");
    ICHECK_NOTNULL(add_op_strategy);

    relay::Var x = relay::Var("x",
                              TensorType({1, 3, 64, 64},
                                         DataType::Float(32)));
//    relay::Constant x = relay::Constant(runtime::NDArray::Empty({1, 3, 64, 64},
//                                                                 {kDLFloat, 32, 1},
//                                                                 {kDLCPU, 0}));
    relay::Constant y = relay::Constant(runtime::NDArray::Empty({1, 3, 64, 64},
                                                                {kDLFloat, 32, 1},
                                                                {kDLCPU, 0}));
    relay::Expr z = BuildAddExpr(x, y);
//    relay::Expr output = BuildConvBNRelu(x, 3, 16, 1, 1, 1, 1, 3);
//    std::string res = relay::AsText(IRModule::FromExpr(output), false);
//    std::cout << res << std::endl;
//    const PackedFunc* flower_call = runtime::Registry::Get("relay.backend.lower_call");
//    ICHECK_NOTNULL(flower_call);
    const PackedFunc* fp = runtime::Registry::Get("relay._transform.FoldConstantExpr");

    relay::Expr after = (*fp)(z, IRModule::FromExpr(z), false);
    std::cout << relay::AsText(IRModule::FromExpr(after), false) << std::endl;
}
