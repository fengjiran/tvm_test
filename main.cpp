//#include <iostream>
#include <string>
#include "tvm/runtime/memory.h"
#include "tvm/runtime/object.h"
#include "tvm/runtime/registry.h"
//#include "tvm/runtime/packed_func.h"

using TVMArgs = tvm::runtime::TVMArgs;
using TVMRetValue = tvm::runtime::TVMRetValue;
using PackedFunc = tvm::runtime::PackedFunc;

void MyAdd(TVMArgs args, TVMRetValue *rv) {
    // automatically convert arguments to desired type.
    int a = args[0];
    int b = args[1];

    // automatically assign value return to rv
    *rv = a + b;
}

void CallPacked() {
    PackedFunc myadd = PackedFunc(MyAdd);
    // get back 3
    int c = myadd(1, 2);
}

int main() {
    TVM_REGISTER_GLOBAL("myadd").set_body(MyAdd);
    PackedFunc myadd = PackedFunc(MyAdd);
    int c = myadd(1, 2);
    std::cout << "myadd = " << c << std::endl;
    const std::string func_name = "runtime.DumpTypeTable";
//    const std::string func_name = "myadd";
    const PackedFunc *fp = tvm::runtime::Registry::Get(func_name);
    ICHECK(fp != nullptr);
    (*fp)(0);
    return 0;
}
