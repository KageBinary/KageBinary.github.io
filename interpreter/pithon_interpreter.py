import sys
from textx import metamodel_from_file
import matplotlib.pyplot as plt
import numpy as np


class Interpreter:
    def __init__(self, code_string, varMap=None, debug=False):
        self.code_string = code_string
        self.varMap = varMap if varMap else {}
        self.debug = debug
        self.built_in_functions = {
            'sin': np.sin,
            'cos': np.cos,
            'tan': np.tan,
            'log': np.log,
            'sqrt': np.sqrt
            # Add more if needed
        }
        self.modules = {
            'Math': {
                'pi': np.pi,
                'e': np.e,
                'tau': 2 * np.pi,
                'phi': (1 + np.sqrt(5)) / 2,
                'gamma': 0.577215664901532,
                'sqrt2': np.sqrt(2),
                'sqrt3': np.sqrt(3),
                'log10e': np.log10(np.e),
            },
            'Physics': {
                'c': 299792458,
                'G': 6.67430e-11,
                'h': 6.62607015e-34,
                'Na': 6.02214076e23,
                'k': 1.380649e-23
            },
            'System': {
                'inf': float('inf'),
                '-inf': float('-inf'),
                'eps': np.finfo(float).eps
            }
        }
        self.imported_modules = {}
        self.output_buffer = []
        self.graph_buffer = []
        self.functionMap = {}
        self.methodMap = {}
        self.meta_model = None
        self.model = None

    def load_meta_model(self, grammar_file="pithon.tx"):
        try:
            self.meta_model = metamodel_from_file(grammar_file, debug=self.debug)
        except Exception as e:
            raise ValueError(f"Error loading grammar: {e}")

    def parse_code(self):
        if not self.meta_model:
            self.load_meta_model()

        try:
            self.model = self.meta_model.model_from_str(self.code_string)
        except Exception as e:
            raise ValueError(f"Error parsing code: {e}")

    def run(self):
        self.parse_code()

        for statement in self.model.statements:
            self.process_statement(statement)

        # Print all buffered outputs
        for output in self.output_buffer:
            if isinstance(output, float):
                output = round(output, 10)  # Round to 10 decimal places
            print(output)

        # Process all graphs after everything else
        for graph in self.graph_buffer:
            self.process_graph_declaration(graph)
        plt.show()


    def infer_type(self, value):
        """
        Infers the type of a value based on Python's data types,
        mapping them to the custom language's types.
        """
        if isinstance(value, int):
            return 'Integers'
        elif isinstance(value, float):
            return 'Reals'
        elif isinstance(value, str):
            return 'Strings'
        elif isinstance(value, bool):
            return 'Booleans'
        elif isinstance(value, list):
            return 'Lists'
        elif isinstance(value, set):
            return 'Sets'
        else:
            raise ValueError(f"Cannot infer type for value: {value}")

    def process_statement(self, statement):
        class_name = statement.__class__.__name__

        if class_name == "ForLoop":
            self.process_for_loop(statement)

        elif class_name == "VariableDeclaration":
            var_name = statement.newVar
            var_value = None
            var_type = None
            if hasattr(statement, 'varValue') and statement.varValue:
                var_value = self.evaluate_operation(statement.varValue)
                inferred_type = self.infer_type(var_value)
                var_type = statement.varType if statement.varType else inferred_type
                if var_type != inferred_type:
                    raise ValueError(f"Type mismatch when declaring '{var_name}': "
                                     f"expected '{var_type}' but got '{inferred_type}'.")
            else:
                var_type = statement.varType
                var_value = None

            if var_name in self.varMap:
                raise ValueError(f"Variable '{var_name}' is already declared.")
            self.varMap[var_name] = {'type': var_type, 'value': var_value}

        elif class_name == "DisplayStatement":
            display_value = self.evaluate_operation(statement.displayValue) if statement.displayValue else None
            if display_value is not None:
                self.output_buffer.append(display_value)

        elif class_name == "Expression":
            # Evaluate the 'left' attribute of the Expression node (OrExpr)
            self.evaluate_operation(statement.left)

        elif class_name == "ImportStatement":
            for module in statement.modules:
                if module in self.modules:
                    self.imported_modules[module] = self.modules[module]
                else:
                    raise ValueError(f"Module '{module}' not found.")

        elif class_name == "WhileLoop":
            self.process_while_loop(statement)

        elif class_name == "AssignmentStatement":
            var_name = statement.varName
            if var_name not in self.varMap:
                raise ValueError(f"Variable '{var_name}' is not declared.")
            var_value = self.evaluate_operation(statement.varValue)
            var_info = self.varMap[var_name]
            inferred_type = self.infer_type(var_value)
            if var_info['type'] != inferred_type:
                raise ValueError(f"Type mismatch: Variable '{var_name}' is of type '{var_info['type']}' "
                                 f"but got '{inferred_type}'.")

            var_info['value'] = var_value

        elif class_name == "GraphDeclaration":
            self.graph_buffer.append(statement)

        elif class_name == "FunctionDeclaration":
            func_name = statement.funcName
            params = statement.params
            body = statement.funcExpr

            if func_name in self.functionMap:
                raise ValueError(f"Function '{func_name}' is already defined.")
            self.functionMap[func_name] = (params, body)

        elif class_name == "FunctionCall":
            # Direct evaluation if it stands alone as a statement
            self.evaluate_operation(statement)

        elif class_name == "SetOperation":
            result = self.evaluate_operation(statement)
            self.output_buffer.append(result)

        elif class_name == "MethodDeclaration":
            self.process_method_declaration(statement)

        elif class_name == "MethodCall":
            result = self.process_method_call(statement)
            if result is not None:
                self.output_buffer.append(result)

        elif class_name == "IfStatement":
            condition_val = self.evaluate_operation(statement.condition)
            if not isinstance(condition_val, bool):
                raise ValueError("If condition must evaluate to a boolean.")

            if condition_val:
                for stmt in statement.ifBody:
                    self.process_statement(stmt)
            else:
                if hasattr(statement, 'elseBody') and statement.elseBody:
                    for stmt in statement.elseBody:
                        self.process_statement(stmt)

        else:
            raise ValueError(f"Unhandled statement type: {class_name}")


    def apply_function(self, func_name, input_set):
        """
        Applies a user-defined function to each element of a set and returns a new set with the results.

        Parameters:
            func_name (str): The name of the function to apply.
            input_set (set): The set of elements to transform.

        Returns:
            set: A new set containing the transformed elements.
        """
        if not isinstance(input_set, set):
            raise TypeError("The second argument to apply must be a set.")
        if not isinstance(func_name, str) or func_name not in self.functionMap:
            raise TypeError("The first argument to apply must be a valid function name.")

        params, body = self.functionMap[func_name]
        if len(params) != 1:
            raise ValueError("Function passed to apply must take exactly one argument.")

        param_name = params[0]
        result_set = set()

        for el in input_set:
            # Backup the current variable map
            previous_var_map = self.varMap.copy()

            # Set the function parameter to the current element
            self.varMap[param_name] = {'type': self.infer_type(el), 'value': el}

            try:
                # Evaluate the function body
                result = self.evaluate_operation(body)
                result_set.add(result)
            except Exception as e:
                raise ValueError(f"Error applying function '{func_name}' to element '{el}': {e}")
            finally:
                # Restore the previous variable map
                self.varMap = previous_var_map

        return result_set
    

    def process_while_loop(self, loop):
        # Evaluate the condition first
        condition_val = self.evaluate_operation(loop.condition)
        # condition_val should be a boolean
        if not isinstance(condition_val, bool):
            raise ValueError("While loop condition must evaluate to a boolean.")

        # Run the loop while condition is True
        while self.evaluate_operation(loop.condition):
            # Save current state of varMap before iteration (if you need to restore after break, etc.)
            previous_var_map = self.varMap.copy()
            try:
                for stmt in loop.body:
                    self.process_statement(stmt)
            except Exception as e:
                # If there's an error, restore old varMap and raise
                self.varMap = previous_var_map
                raise ValueError(f"Error in while loop body execution: {e}")
            else:
                # After successful iteration, keep varMap changes
                pass

    def process_for_loop(self, loop):
        loop_var = loop.loopVar
        start = self.evaluate_operation(loop.start)
        end = self.evaluate_operation(loop.end)
        step = self.evaluate_operation(loop.step) if loop.step else 1

        current_value = start

        while (current_value < end if step > 0 else current_value > end):
            previous_var_map = self.varMap.copy()
            self.varMap[loop_var] = {'type': 'Integers', 'value': current_value}

            try:
                for stmt in loop.body:
                    self.process_statement(stmt)
            except Exception as e:
                self.varMap = previous_var_map
                raise ValueError(f"Error in loop body execution: {e}")
            else:
                self.varMap = previous_var_map

            current_value += step

    def extract_loop_body_as_string(self, body_statements):
        body_strings = []
        for stmt in body_statements:
            body_strings.append(str(stmt))
        return "\n".join(body_strings)

    def process_graph_declaration(self, graph):
        x_start = self.evaluate_operation(graph.xStart)
        x_end = self.evaluate_operation(graph.xEnd)
        x_step = self.evaluate_operation(graph.xStep)
        y_start = self.evaluate_operation(graph.yStart)
        y_end = self.evaluate_operation(graph.yEnd)
        y_func_call = graph.yFunc
        style = graph.style.strip('"')

        y_func = y_func_call.funcName
        if y_func not in self.functionMap:
            raise ValueError(f"Function '{y_func}' is not defined.")

        params, body = self.functionMap[y_func]

        if len(params) != 1:
            raise ValueError(f"Graph function '{y_func}' must take exactly one parameter.")

        x_values_smooth = np.linspace(x_start, x_end, 1000)
        y_values_smooth = []

        for x in x_values_smooth:
            self.varMap[params[0]] = {'type': 'Reals', 'value': x}
            y_val = self.evaluate_operation(body)
            y_values_smooth.append(y_val)

        x_range = x_end - x_start
        y_range = y_end - y_start

        if x_step <= 0 or x_step > x_range / 5:
            x_step = x_range / 10
        elif x_step < x_range / 100:
            x_step = x_range / 50

        y_step = self.evaluate_operation(graph.yStep)
        if y_step <= 0 or y_step > y_range / 5:
            y_step = y_range / 10
        elif y_step < y_range / 100:
            y_step = y_range / 50

        x_major_step = x_step * (10 if x_step < x_range / 20 else 5)
        y_major_step = y_step * (10 if y_step < y_range / 20 else 5)

        plt.figure()
        plt.plot(x_values_smooth, y_values_smooth, style, label=f"{y_func}(x)")

        x_minor_ticks = np.arange(x_start, x_end + x_step, x_step)
        x_major_ticks = np.arange(x_start, x_end + x_major_step, x_major_step)
        y_minor_ticks = np.arange(y_start, y_end + y_step, y_step)
        y_major_ticks = np.arange(y_start, y_end + y_major_step, y_major_step)

        plt.xticks(x_major_ticks)
        plt.yticks(y_major_ticks)
        plt.minorticks_on()
        plt.gca().set_xticks(x_minor_ticks, minor=True)
        plt.gca().set_yticks(y_minor_ticks, minor=True)

        plt.xlim(x_start, x_end)
        plt.ylim(y_start, y_end)

        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.grid(True, which='major', color='black', linewidth=1)

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f"Graph of {y_func}(x)")
        plt.legend()

    def process_method_declaration(self, statement):
        method_name = statement.methodName
        return_type = statement.returnType
        params = [(param.paramType, param.paramName) for param in statement.params]
        body = statement.body

        if method_name in self.methodMap:
            raise ValueError(f"Method '{method_name}' is already defined.")

        self.methodMap[method_name] = {
            'returnType': return_type,
            'params': params,
            'body': body
        }

    def process_method_call(self, statement):
        if statement.object:
            obj = self.evaluate_operation(statement.object)
            method_name = statement.methodName
            args = [self.evaluate_operation(a) for a in statement.args]

            if isinstance(obj, dict) and method_name in obj and callable(obj[method_name]):
                return obj[method_name](*args)
            else:
                raise ValueError(f"Object '{statement.object}' has no accessible method '{method_name}'.")
        else:
            method_name = statement.methodName
            if method_name not in self.methodMap:
                raise ValueError(f"Method '{method_name}' is not defined.")

            method_details = self.methodMap[method_name]
            expected_params = method_details['params']
            body = method_details['body']
            return_type = method_details['returnType']

            if len(expected_params) != len(statement.args):
                raise ValueError(
                    f"Method '{method_name}' expects {len(expected_params)} arguments, "
                    f"but got {len(statement.args)}."
                )

            previous_var_map = self.varMap.copy()
            try:
                for (expected_type, param_name), arg_expr in zip(expected_params, statement.args):
                    arg_value = self.evaluate_operation(arg_expr)
                    arg_type = self.infer_type(arg_value)
                    if expected_type == 'Reals':
                        if arg_type == 'Integers':
                            arg_type = 'Reals'
                    if arg_type != expected_type:
                        raise ValueError(
                            f"Argument '{param_name}' for method '{method_name}' expects type '{expected_type}' "
                            f"but got '{arg_type}'."
                        )
                    self.varMap[param_name] = {'type': expected_type, 'value': arg_value}

                result = None
                for stmt in body:
                    result = self.process_statement(stmt)

                if result is not None:
                    res_type = self.infer_type(result)
                    if res_type != return_type:
                        raise ValueError(
                            f"Method '{method_name}' expected to return type '{return_type}' "
                            f"but got '{res_type}'."
                        )
                return result
            finally:
                self.varMap = previous_var_map

    def process_function_call(self, func_call_expr):
        func_name = func_call_expr.funcName

        # First, check built-in functions
        if func_name in self.built_in_functions:
            args_values = [self.evaluate_operation(arg) for arg in func_call_expr.args]
            return self.built_in_functions[func_name](*args_values)

        # If not built-in, fall back to user-defined
        if func_name not in self.functionMap:
            raise ValueError(f"Function '{func_name}' is not defined.")

        params, body = self.functionMap[func_name]
        if len(params) != len(func_call_expr.args):
            raise ValueError(
                f"Function '{func_name}' expects {len(params)} arguments, but got {len(func_call_expr.args)}."
            )

        previous_var_map = self.varMap.copy()
        try:
            for param_name, arg_expr in zip(params, func_call_expr.args):
                arg_value = self.evaluate_operation(arg_expr)
                self.varMap[param_name] = {'type': self.infer_type(arg_value), 'value': arg_value}

            result = self.evaluate_operation(body)
            return result
        finally:
            self.varMap = previous_var_map


    def evaluate_operation(self, expr):
        if expr is None:
            return None

        if isinstance(expr, (int, float)):
            return expr
        if isinstance(expr, str):
            if expr in self.varMap:
                return self.varMap[expr]['value']
            return expr

        class_name = expr.__class__.__name__

        if class_name == "Expression":
            return self.evaluate_operation(expr.left)

        elif class_name == "OrExpr":
            result = self.evaluate_operation(expr.left)
            for right_expr in expr.rights:
                right_val = self.evaluate_operation(right_expr)
                result = result or right_val
            return result

        elif class_name == "AndExpr":
            result = self.evaluate_operation(expr.left)
            for right_expr in expr.rights:
                right_val = self.evaluate_operation(right_expr)
                result = result and right_val
            return result

        elif class_name == "EqExpr":
            left_val = self.evaluate_operation(expr.left)
            if hasattr(expr, 'op') and expr.op is not None:
                right_val = self.evaluate_operation(expr.right)
                if expr.op == '==':
                    return left_val == right_val
                elif expr.op == '!=':
                    return left_val != right_val
            return left_val

        elif class_name == "Comparison":
            left_val = self.evaluate_operation(expr.left)
            if expr.comp and expr.right:
                right_val = self.evaluate_operation(expr.right)
                if expr.comp == '==':
                    return left_val == right_val
                elif expr.comp == '!=':
                    return left_val != right_val
                elif expr.comp == '<':
                    return left_val < right_val
                elif expr.comp == '<=':
                    return left_val <= right_val
                elif expr.comp == '>':
                    return left_val > right_val
                elif expr.comp == '>=':
                    return left_val >= right_val
            else:
                return left_val

        elif class_name == "Addition":
            left_val = self.evaluate_operation(expr.left)
            for op, right in zip(expr.ops, expr.rights):
                right_val = self.evaluate_operation(right)
                if op == '+':
                    if isinstance(left_val, str) or isinstance(right_val, str):
                        left_val = str(left_val) + str(right_val)
                    else:
                        left_val = left_val + right_val
                elif op == '-':
                    if isinstance(left_val, (int, float)) and isinstance(right_val, (int, float)):
                        left_val = left_val - right_val
                    else:
                        raise ValueError(
                            f"Subtraction operator '-' not supported for types {type(left_val).__name__} and {type(right_val).__name__}."
                        )
            return left_val

        elif class_name == "ApplyFunction":
            func = expr.func
            input_set = self.evaluate_operation(expr.inputSet)
            return self.apply_function(func, input_set)
        elif class_name == "Multiplication":
            result = self.evaluate_operation(expr.left)
            for op, right in zip(expr.ops, expr.rights):
                right_val = self.evaluate_operation(right)
                if op == '*':
                    result = result * right_val
                else:
                    if right_val == 0:
                        raise ValueError("Division by zero.")
                    result = result / right_val
            return result

        elif class_name == "Exponentiation":
            result = self.evaluate_operation(expr.left)
            for right in expr.rights:
                right_val = self.evaluate_operation(right)
                result = result ** right_val
            return result

        elif class_name == "Primary":
            if expr.op == '-':
                return -self.evaluate_operation(expr.prim)
            elif expr.op == '!':
                # If we have unary '!', it should have been integrated into higher-level boolean rules.
                # If kept here, handle negation if needed. Otherwise, remove this or handle logically above.
                val = self.evaluate_operation(expr.prim)
                if not isinstance(val, bool):
                    raise ValueError("Logical negation '!' used on non-boolean value.")
                return not val
            return self.evaluate_operation(expr.prim)

        elif class_name == "MethodInvocation":
            return self.process_method_invocation(expr)
        elif class_name == "Number":
            return expr.value

        elif class_name == "Summation":
            return self.process_summation(expr)

        elif class_name == "StringLiteral":
            return expr.string.strip('"')

        elif class_name == "Variable":
            var_name = expr.var
            if '.' in var_name:
                parts = var_name.split('.')
                current = self.imported_modules
                for part in parts:
                    if part in current:
                        current = current[part]
                    else:
                        raise ValueError(f"'{part}' not found in the module.")
                return current
            else:
                if var_name in self.varMap:
                    return self.varMap[var_name]['value']
                raise ValueError(f"Variable '{var_name}' is not defined.")

        elif class_name == "SetLiteral":
            elements = [self.evaluate_operation(e) for e in expr.elements]
            element_types = {self.infer_type(el) for el in elements}
            if len(element_types) > 1:
                raise ValueError(f"Sets must contain elements of the same type, got multiple: {element_types}")
            return set(elements)

        elif class_name == "SetOperation":
            return self.process_set_operation(expr)

        elif class_name == "FunctionCall":
            return self.process_function_call(expr)

        elif class_name == "Factorial":
            return self.process_factorial(expr)

        elif class_name == "MethodCall":
            return self.process_method_call(expr)

        elif class_name == "BooleanLiteral":
            return True if expr.value == 'true' else False

        raise ValueError(f"Unhandled expression type: {class_name}")

    def process_factorial(self, factorial_expr):
        val = self.evaluate_operation(factorial_expr.expr)
        if not isinstance(val, int):
            raise ValueError("Factorial argument must be an integer.")
        if val < 0:
            raise ValueError("Factorial is not defined for negative integers.")

        result = 1
        for i in range(1, val + 1):
            result *= i
        return result

    def process_method_invocation(self, invocation_expr):
        # If there's an object, it means something like obj.methodName(args)
        if invocation_expr.object:
            obj = self.evaluate_operation(invocation_expr.object)
            method_name = invocation_expr.methodName
            args = [self.evaluate_operation(a) for a in invocation_expr.args]

            # If obj is a module or dict (like imported modules), try to find a callable
            if isinstance(obj, dict) and method_name in obj and callable(obj[method_name]):
                return obj[method_name](*args)
            else:
                # If it's not a module or doesn't have a method, raise an error.
                # If you want other object types, handle them here.
                raise ValueError(f"Object '{invocation_expr.object.var}' has no accessible method '{method_name}'.")
        else:
            # No object given, meaning a method is invoked as if it were standalone.
            # This implies we are calling a 'method' declared via MethodDeclaration.
            method_name = invocation_expr.methodName
            if method_name not in self.methodMap:
                raise ValueError(f"Method '{method_name}' is not defined.")

            method_details = self.methodMap[method_name]
            params = method_details['params']
            body = method_details['body']

            if len(params) != len(invocation_expr.args):
                raise ValueError(
                    f"Method '{method_name}' expects {len(params)} arguments, "
                    f"but got {len(invocation_expr.args)}."
                )

            previous_var_map = self.varMap.copy()
            try:
                # Bind arguments to parameters
                for (expected_type, param_name), arg_expr in zip(params, invocation_expr.args):
                    arg_value = self.evaluate_operation(arg_expr)
                    arg_type = self.infer_type(arg_value)
                    # Handle Reals/Integers conversion if needed
                    if expected_type == 'Reals' and arg_type == 'Integers':
                        arg_type = 'Reals'
                    if arg_type != expected_type:
                        raise ValueError(
                            f"Argument '{param_name}' for method '{method_name}' expects type '{expected_type}' "
                            f"but got '{arg_type}'."
                        )
                    self.varMap[param_name] = {'type': expected_type, 'value': arg_value}

                # Execute method body and get return value
                result = None
                for stmt in body:
                    result = self.process_statement(stmt)

                # Check return type
                if self.infer_type(result) != method_details['returnType']:
                    raise ValueError(
                        f"Method '{method_name}' expected to return type '{method_details['returnType']}' "
                        f"but got '{self.infer_type(result)}'."
                    )

                return result
            finally:
                # Restore previous var map
                self.varMap = previous_var_map


    def process_summation(self, summation_expr):
        var_name = summation_expr.varName
        start = self.evaluate_operation(summation_expr.start)
        end = self.evaluate_operation(summation_expr.end)

        if not isinstance(start, int) or not isinstance(end, int):
            raise ValueError("Summation start and end must be integers.")

        total = 0
        previous_var_map = self.varMap.copy()
        try:
            for i in range(start, end + 1):
                self.varMap[var_name] = {'type': 'Integers', 'value': i}
                val = self.evaluate_operation(summation_expr.expr)
                if not isinstance(val, (int, float)):
                    raise ValueError("Summation expression must evaluate to a number.")
                total += val
        finally:
            self.varMap = previous_var_map

        return total

    def process_set_operation(self, operation):
        result = self.evaluate_operation(operation.left)
        for op, right in zip(operation.ops, operation.rights):
            right_val = self.evaluate_operation(right)
            if not isinstance(result, set) or not isinstance(right_val, set):
                raise ValueError("Set operations require set operands.")
            if op == "<+>":
                result = result.union(right_val)
            elif op == "<->":
                result = result.difference(right_val)
            elif op == "<*>":
                result = result.intersection(right_val)
            elif op == "</>":
                result = result.symmetric_difference(right_val)
        return result


def extract_code_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        raise ValueError(f"File '{file_path}' not found.")


def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <input_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    code = extract_code_from_file(file_path)

    interpreter = Interpreter(code_string=code, debug=False)
    interpreter.run()


if __name__ == "__main__":
    main()
