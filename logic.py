import nltk
import re
from nltk.sem import Expression
from nltk.inference import ResolutionProver

class LogicEngine:
    def __init__(self, kb_file):
        self.kb_file = kb_file
        self.kb_expressions = []
        self.read_expr = Expression.fromstring
        self.load_kb()

    def load_kb(self):
        try:
            with open(self.kb_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if line:
                        try:
                            expr = self.read_expr(line)
                            self.kb_expressions.append(expr)
                        except Exception as e:
                            print(f"Error parsing KB line '{line}': {e}")
            

            if  ResolutionProver().prove(self.read_expr('P(x) & -P(x)'), self.kb_expressions, verbose=False):
                print("Warning: Initial KB contains contradictions!")
            else:
                print("KB loaded successfully, no initial contradictions found.")

        except FileNotFoundError:
            print("KB file not found, starting with empty KB.")

    def add_knowledge(self, subject, property_name):



        subj = subject.capitalize()
        prop = property_name.capitalize()
        new_stmt_str = f"{prop}({subj})"
        
        try:
            new_expr = self.read_expr(new_stmt_str)

            neg_expr = self.read_expr(f"-{new_stmt_str}")
            
            if ResolutionProver().prove(neg_expr, self.kb_expressions, verbose=False):
                return f"I cannot add that, it contradicts my existing knowledge."
            

            self.kb_expressions.append(new_expr)
            with open(self.kb_file, 'a') as f:
                f.write(f"\n{new_stmt_str}")
                
            return f"OK, I will remember that {subject} is {property_name}."
            
        except Exception as e:
            return f"Error processing new knowledge: {e}"

    def check_knowledge(self, subject, property_name):
        subj = subject.capitalize()
        prop = property_name.capitalize()
        
        try:
            query = self.read_expr(f"{prop}({subj})")
            neg_query = self.read_expr(f"-{prop}({subj})")
            
            is_true = ResolutionProver().prove(query, self.kb_expressions, verbose=False)
            is_false = ResolutionProver().prove(neg_query, self.kb_expressions, verbose=False)
            
            if is_true:
                return "Correct."
            elif is_false:
                return "Incorrect."
            else:
                return "I don't know."
        except Exception as e:
            return f"Error verifying knowledge: {e}"

    def add_multivalued(self, subject, verb, object_name):
        subj = subject.capitalize()
        v = verb.capitalize()
        obj = object_name.capitalize()
        new_stmt_str = f"{v}({subj}, {obj})"
        
        try:
            new_expr = self.read_expr(new_stmt_str)
            neg_expr = self.read_expr(f"-{new_stmt_str}")
            
            if ResolutionProver().prove(neg_expr, self.kb_expressions, verbose=False):
                return f"I cannot add that, it contradicts my existing knowledge."
            
            self.kb_expressions.append(new_expr)
            with open(self.kb_file, 'a') as f:
                f.write(f"\n{new_stmt_str}")
                
            return f"OK, I will remember that {subject} {verb} {object_name}."
            
        except Exception as e:
            return f"Error processing new knowledge: {e}"

    def check_multivalued(self, subject, verb, object_name):
        subj = subject.capitalize()
        v = verb.capitalize()
        obj = object_name.capitalize()
        
        try:
            query = self.read_expr(f"{v}({subj}, {obj})")
            neg_query = self.read_expr(f"-{v}({subj}, {obj})")
            
            is_true = ResolutionProver().prove(query, self.kb_expressions, verbose=False)
            is_false = ResolutionProver().prove(neg_query, self.kb_expressions, verbose=False)
            
            if is_true:
                return "Correct."
            elif is_false:
                return "Incorrect."
            else:
                return "I don't know."
        except Exception as e:
            return f"Error verifying knowledge: {e}"

if __name__ == "__main__":
    le = LogicEngine("kb.txt")
    print(le.check_knowledge("Rex", "Dog"))
    print(le.add_knowledge("Buddy", "Dog"))
    print(le.check_knowledge("Buddy", "Animal"))
