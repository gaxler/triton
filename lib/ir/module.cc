#include <algorithm>
#include <iostream>
#include "triton/ir/basic_block.h"
#include "triton/ir/module.h"
#include "triton/ir/type.h"
#include "triton/ir/constant.h"
#include "triton/ir/function.h"

namespace triton{
namespace ir{

/* functions */
function *module::get_or_insert_function(const std::string &name, function_type *ty) {
  function *&fn = (function*&)symbols_[name];
  if(fn == nullptr)
    return fn = function::create(ty, global_value::external, name, this);
  return fn;
}


}
}
