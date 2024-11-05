## Guide to Registering a New Operator in Op Builder Registry

To register a new operator in op builder registry, please follow below steps:

1. Open the op builder registry located in the directory /src/ops/op_registration_db.def.
2. Choose from one of these three registration mechanisms to fits specific needs:

    i. `reg_op_with_def_policy`: If we want to register a new op with the default policy, this mechanism is a perfect choice.</br>
    ii. `reg_op_with_def_policy_wo_attr`: If we want to skip passing attributes while creating objects, choose this option.</br>
    iii. `reg_op_with_custom_policy`: If we want to specify a custom object creation logic, this mechanism is the right choice.</br>

3. we will need to provide the following arguments, regardless of what mechanism we choose:

    i. The name of op.</br>
    ii. Type vectors (a_type, b_type, c_type, and so on) to indicate the types of op's attributes.</br>
    iii. Positions at which these types are present in attributes. If we want to skip validation for certain types, positions can be set to -1. If we want to skip validation entirely, pass an empty vector {}.</br>

That’s it! Follow these simple steps, and new operator will be registered in op builder registry.

#### Examples: https://confluence.amd.com/display/XSW/Operator+Migration+Examples

We can create multiple rules for the same operation type. The final output will be determined by the rule that follows any validation criteria that we have set. We have the ability to skip the validation process for individual operations, but only once per operation. If we skip the validation process for an operation more than once, the previous skipped validation will be replaced with the current one.
