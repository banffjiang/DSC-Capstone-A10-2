import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataloader import PassageQuestionLoader

def main():
    loader = PassageQuestionLoader("data/passage_question_train.jsonl")

    for example in loader:
        passage = example["passage"]
        question = example["question"]
        print("PASSAGE:", passage)
        print("QUESTION:", question)
        break  #just tesitng first one
if __name__ == "__main__":
    main()
