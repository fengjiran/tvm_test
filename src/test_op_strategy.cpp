//
// Created by 赵丹 on 2023/8/17.
//
#include "tvm/relay/expr.h"
#include "tvm/relay/op.h"
#include "tvm/relay/op_strategy.h"
#include "tvm/topi/generic/injective.h"
#include "tvm/topi/broadcast.h"

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

void ResetAddOpStrategy() {
    auto reg_op_attr = runtime::Registry::Get("ir.RegisterOpAttr");
    ICHECK_NOTNULL(reg_op_attr);

    auto reset_op_attr = runtime::Registry::Get("ir.OpResetAttr");
    ICHECK_NOTNULL(reset_op_attr);

    auto add_op_strategy = runtime::Registry::Get("test.add_strategy");
    ICHECK_NOTNULL(add_op_strategy);

    auto fgeneric = GenericFunc::Get("test.strategy_generic").set_default(*add_op_strategy, true);
    auto add_op = relay::Op::Get("add");
    (*reset_op_attr)(add_op, "FTVMStrategy");
    (*reg_op_attr)("add", "FTVMStrategy", fgeneric, 10);
}