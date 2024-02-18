from sympy import simplify, sympify, Rational


def symbol_calculate(expression):
    """
    We use this function to calculate the result of a symbolic expression.
    :param expression: a string of symbolic expression
    :return: a string of the result
    """
    try:
        expr = sympify(expression)
        result = simplify(expr)

        # Check if the result is a number
        if result.is_number:
            # If the result is a rational number, return as a fraction
            if isinstance(result, Rational):
                return str(result)
            # If the result is a floating point number, format to remove redundant zeros
            else:
                return "{:g}".format(float(result))
        else:
            return str(result)
    except Exception as e:
        return f"Error: {str(e)}"