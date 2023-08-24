//
// Created by 赵丹 on 2023/8/17.
//
#include "tvm/relay/expr.h"
#include "tvm/relay/op.h"
#include "tvm/relay/op_strategy.h"
#include "tvm/relay/attrs/transform.h"
#include "tvm/topi/generic/injective.h"
#include "tvm/topi/generic/default.h"
#include "tvm/topi/broadcast.h"
#include "tvm/topi/nn.h"
#include "test_op_strategy.h"

using namespace tvm;
using namespace tvm::relay;

TVM_REGISTER_GLOBAL("relay.backend.lower_call")
.set_body_typed([](const relay::Call &call, const Array<te::Tensor> &inputs,
                   const Target &target) {
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

TVM_REGISTER_GLOBAL("test.add_strategy")
.set_body_typed([](const Attrs &attrs, const Array<te::Tensor> &inputs, const Type &out_type,
                   const Target &target) {
    FTVMCompute fcompute = [](const Attrs &attrs, const Array<te::Tensor> &inputs,
                              const Type &out_type) -> Array<te::Tensor> {
        ICHECK_EQ(inputs.size(), 2U);
        return {topi::add(inputs[0], inputs[1])};
    };
    FTVMSchedule fschedule = [](const Attrs &attrs, const Array<te::Tensor> &outs,
                                const Target &target) {
        With<Target> target_scope(target);
//        return topi::generic::schedule_injective(target, outs);
        return topi::generic::default_schedule(target, outs);
    };

    auto n = make_object<OpStrategyNode>();
    auto strategy = relay::OpStrategy(std::move(n));
    strategy.AddImplementation(fcompute, fschedule, "test.add_strategy", 10);
    return strategy;
});

TVM_REGISTER_GLOBAL("test.nn.relu_strategy")
.set_body_typed([](const Attrs &attrs, const Array<te::Tensor> &inputs, const Type &out_type,
                   const Target &target) {
    FTVMCompute fcompute = [](const Attrs &attrs, const Array<te::Tensor> &inputs,
                              const Type &out_type) -> Array<te::Tensor> {
        ICHECK_EQ(inputs.size(), 1U);
        return {topi::relu(inputs[0], 0.0f)};
    };
    FTVMSchedule fschedule = [](const Attrs &attrs, const Array<te::Tensor> &outs,
                                const Target &target) {
        With<Target> target_scope(target);
//        return topi::generic::schedule_injective(target, outs);
        return topi::generic::default_schedule(target, outs);
    };

    auto n = make_object<OpStrategyNode>();
    auto strategy = relay::OpStrategy(std::move(n));
    strategy.AddImplementation(fcompute, fschedule, "test.nn.relu_strategy", 10);
    return strategy;
});

TVM_REGISTER_GLOBAL("test.multiply_strategy")
.set_body_typed([](const Attrs &attrs, const Array<te::Tensor> &inputs, const Type &out_type,
                   const Target &target) {
    FTVMCompute fcompute = [](const Attrs &attrs, const Array<te::Tensor> &inputs,
                              const Type &out_type) -> Array<te::Tensor> {
        ICHECK_EQ(inputs.size(), 2U);
        return {topi::multiply(inputs[0], inputs[1])};
    };
    FTVMSchedule fschedule = [](const Attrs &attrs, const Array<te::Tensor> &outs,
                                const Target &target) {
        With<Target> target_scope(target);
//        return topi::generic::schedule_injective(target, outs);
        return topi::generic::default_schedule(target, outs);
    };

    auto n = make_object<OpStrategyNode>();
    auto strategy = relay::OpStrategy(std::move(n));
    strategy.AddImplementation(fcompute, fschedule, "test.multiply_strategy", 10);
    return strategy;
});

TVM_REGISTER_GLOBAL("test.concatenate_strategy")
.set_body_typed([](const Attrs &attrs, const Array<te::Tensor> &inputs, const Type &out_type,
                   const Target &target) {
    FTVMCompute fcompute = [](const Attrs &attrs, const Array<te::Tensor> &inputs,
                              const Type &out_type) -> Array<te::Tensor> {
        const auto *param = attrs.as<ConcatenateAttrs>();
        ICHECK(param != nullptr);
        return {topi::concatenate(inputs, param->axis)};
    };

    FTVMSchedule fschedule = [](const Attrs &attrs, const Array<te::Tensor> &outs,
                                const Target &target) {
        With<Target> target_scope(target);
//        return topi::generic::schedule_injective(target, outs);
        return topi::generic::default_schedule(target, outs);
    };

    auto n = make_object<OpStrategyNode>();
    auto strategy = relay::OpStrategy(std::move(n));
    strategy.AddImplementation(fcompute, fschedule, "test.concatenate_strategy", 10);
    return strategy;


});

void ResetOpStrategy(const std::string &op_name) {
    auto reg_op_attr = runtime::Registry::Get("ir.RegisterOpAttr");
    ICHECK_NOTNULL(reg_op_attr);

    auto reset_op_attr = runtime::Registry::Get("ir.OpResetAttr");
    ICHECK_NOTNULL(reset_op_attr);

    auto op_strategy = runtime::Registry::Get("test." + op_name + "_strategy");
    ICHECK_NOTNULL(op_strategy);

    auto fgeneric = GenericFunc::Get("test." + op_name + "_generic_strategy")
            .set_default(*op_strategy, true);
    auto op = relay::Op::Get(op_name);
    (*reset_op_attr)(op, "FTVMStrategy");
    (*reg_op_attr)(op_name, "FTVMStrategy", fgeneric, 10);
}

//void ResetAddOpStrategy() {
//    auto reg_op_attr = runtime::Registry::Get("ir.RegisterOpAttr");
//    ICHECK_NOTNULL(reg_op_attr);
//
//    auto reset_op_attr = runtime::Registry::Get("ir.OpResetAttr");
//    ICHECK_NOTNULL(reset_op_attr);
//
//    auto add_op_strategy = runtime::Registry::Get("test.add_strategy");
//    ICHECK_NOTNULL(add_op_strategy);
//
//    auto fgeneric = GenericFunc::Get("test.add_generic_strategy").set_default(*add_op_strategy, true);
//    auto add_op = relay::Op::Get("add");
//    (*reset_op_attr)(add_op, "FTVMStrategy");
//    (*reg_op_attr)("add", "FTVMStrategy", fgeneric, 10);
//}

//void ResetReluOpStrategy() {
//    auto reg_op_attr = runtime::Registry::Get("ir.RegisterOpAttr");
//    ICHECK_NOTNULL(reg_op_attr);
//
//    auto reset_op_attr = runtime::Registry::Get("ir.OpResetAttr");
//    ICHECK_NOTNULL(reset_op_attr);
//
//    auto relu_op_strategy = runtime::Registry::Get("test.relu_strategy");
//    ICHECK_NOTNULL(relu_op_strategy);
//
//    auto fgeneric = GenericFunc::Get("test.relu_generic_strategy").set_default(*relu_op_strategy, true);
//    auto relu_op = relay::Op::Get("nn.relu");
//    (*reset_op_attr)(relu_op, "FTVMStrategy");
//    (*reg_op_attr)("nn.relu", "FTVMStrategy", fgeneric, 10);
//}

//void ResetMultiplyOpStrategy() {
//    auto reg_op_attr = runtime::Registry::Get("ir.RegisterOpAttr");
//    ICHECK_NOTNULL(reg_op_attr);
//
//    auto reset_op_attr = runtime::Registry::Get("ir.OpResetAttr");
//    ICHECK_NOTNULL(reset_op_attr);
//
//    auto multiply_op_strategy = runtime::Registry::Get("test.multiply_strategy");
//    ICHECK_NOTNULL(multiply_op_strategy);
//
//    auto fgeneric = GenericFunc::Get("test.multiply_generic_strategy").set_default(*multiply_op_strategy, true);
//    auto multiply_op = relay::Op::Get("multiply");
//    (*reset_op_attr)(multiply_op, "FTVMStrategy");
//    (*reg_op_attr)("multiply", "FTVMStrategy", fgeneric, 10);
//}
