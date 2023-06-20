# This file is part of the Discourse Reasoning project.
# The verifiers are used to measure whether the explanations satisfy the discourse and temporal constraints.
# The verifiers are based on the Z3 theorem prover.

from z3 import Bool, solve, Implies, Not, Or, Solver, And, Real



def add_fol_logical_expression_from_tuple(tuple, variable_dict, solver):

    head = tuple[0]
    tail = tuple[1]

    head_variable = variable_dict[head]
    tail_variable = variable_dict[tail]

    if tuple[2] in  ["Synchronous", "Synchronous inversed", 
                     "Precedence", "Precedence inversed",
                     "Succession", "Succession inversed",
                     "Contrast", "Contrast inversed",
                     "Concession", "Concession inversed",
                     "Conjunction", "Conjunction inversed",
                     "Instantiation", "Instantiation inversed"]:
        
        solver.add(head_variable, tail_variable)
    
    elif tuple[2] in ["Reason", "Condition", "Result inversed"]:
        solver.add( Implies(tail_variable, head_variable))
    
    elif tuple[2] in ["Reason inversed", "Condition inversed", "Result"]:
        solver.add( Implies(head_variable, tail_variable))

    elif tuple[2] in ["Alternative", "Alternative inversed"]:
        solver.add( Or(head_variable, tail_variable))
    
    elif tuple[2] in ["Restatement", "Restatement inversed"]:
        solver.add( head_variable == tail_variable)
    
    elif tuple[2] in ["ChosenAlternative"]:
        solver.add( And(head_variable, Not(tail_variable)))

    elif tuple[2] in ["ChosenAlternative inversed"]:
        solver.add( And(Not(head_variable), tail_variable))
    
    elif tuple[2] in ["Exception"]:
        solver.add(Not(head_variable), tail_variable, Implies(Not(tail_variable), head_variable))
    
    elif tuple[2] in ["Exception inversed"]:
        solver.add(head_variable, Not(tail_variable), Implies(tail_variable, Not(head_variable)))

    else:
        print("Error: unknown tuple type: " + tuple[2])


def add_temporal_logical_expression_from_tuple(tuple, variable_dict, solver):
    
        head = tuple[0]
        tail = tuple[1]
    
        head_variable = variable_dict[head]
        tail_variable = variable_dict[tail]

        all_discourse_types = ["Synchronous", "Synchronous inversed", 
                            "Precedence", "Precedence inversed",
                            "Succession", "Succession inversed",
                            "Contrast", "Contrast inversed",
                            "Concession", "Concession inversed",
                            "Conjunction", "Conjunction inversed",
                            "Instantiation", "Instantiation inversed",
                            "Reason", "Condition", "Result inversed",
                            "Reason inversed", "Condition inversed", "Result",
                            "Alternative", "Alternative inversed",
                            "Restatement", "Restatement inversed",
                            "ChosenAlternative",
                            "ChosenAlternative inversed",
                            "Exception",
                            "Exception inversed"]

        assert tuple[2] in all_discourse_types
    
        if tuple[2] in  ["Synchronous", "Synchronous inversed"]:
            
            solver.add(head_variable == tail_variable)
        
        elif tuple[2] in ["Precedence", "Succession inversed", "Result", "Reason inversed", "Condition inversed"]:
            solver.add( head_variable < tail_variable)
        
        elif tuple[2] in ["Succession", "Precedence inversed", "Reason", "Result inversed", "Condition"]:
            solver.add( head_variable > tail_variable)

        
     
        


def verify_discourse(explanation_tuples):

    variable_dict = {}

    for tuple in explanation_tuples:
        for i in range(2):
            if tuple[i] not in variable_dict:
                variable_dict[tuple[i]] = Bool(tuple[i])

    s = Solver()

    for tuple in explanation_tuples:
        add_fol_logical_expression_from_tuple(tuple, variable_dict, s)
    
    # print(s.check())
    # print(s)

    return s.check().r


def verify_temporal(explanation_tuples):
    variable_dict = {}

    for tuple in explanation_tuples:
        for i in range(2):
            if tuple[i] not in variable_dict:
                variable_dict[tuple[i]] = Real(tuple[i])

    s = Solver()

    for tuple in explanation_tuples:
        add_temporal_logical_expression_from_tuple(tuple, variable_dict, s)
    
    # print(s.check())
    # print(s)

    return s.check().r
    

if __name__ == "__main__":

    example_query_dict = {"query": "(p,(Conjunction inversed),(p,(ChosenAlternative),(e,(she really said this))))", 
        "nl_query": "she really said this, instead of V1. V0, and V1. What is V0?", 
        "train_answers": ["i 'm bluffing"], 
        "train_explanations": [["she really said this, instead of the rest is history.", "i 'm bluffing, and the rest is history."]], 
        "train_explanation_tuples": [[["she really said this", "the rest is history", "ChosenAlternative"], ["the rest is history", "i 'm bluffing", "Conjunction inversed"]]]}
    


    
    # example_tuple_list = [["he learned", "i loved it", "Contrast"], 
    #                     ["you loved it", "it came out", "Synchronous"], 
    #                     ["it came out", "i loved it", "Synchronous inversed"], 
    #                     ["i loved it", "it was amazing", "Conjunction inversed"]]


    # example_contradicton_tuple_list = [["he loved it", "i loved it", "ChosenAlternative"],
    #                                     ["i loved it", "he loved it", "Synchronous"]]



    # example_tempral_violation_tuple_list = [  ["he drink beer", "he drive", "Precedence"],
    #                                         ["he drive", "he play","Precedence"],
    #                                         ["he drink beer", "he play","Succession"] ]

    example_tuple_list = example_query_dict["train_explanation_tuples"][0]


    print(verify_discourse(example_tuple_list))
    # print(verify_discourse(example_contradicton_tuple_list))


    # print(verify_temporal(example_tuple_list))
    # print(verify_temporal(example_tempral_violation_tuple_list))



    

    