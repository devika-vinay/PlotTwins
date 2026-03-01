import ast

def parse_list(x):
        try:
            return ast.literal_eval(x)
        except:
            return []