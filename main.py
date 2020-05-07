from plan import Planner
from new_generator import Generator
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

# For testing purpose.
if __name__ == '__main__':
    planner = Planner()
    generator = Generator()
    while True:
        hint = input("Type first sentence: ")
        keywords = planner.plan(hint)
        print("Keywords: ", keywords)
        poem = generator.generate(keywords, hint)
        print("Poem: ", poem)
