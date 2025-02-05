from typing import List, Tuple


class PromptTemplate:
    def __init__(self, tokenizer, train=True):
        self.tokenizer = tokenizer
        self.train = train

    def get_prompt(self, batch) -> Tuple[List[str], List[str]]:
        raise NotImplementedError("get_prompt method must be implemented in child class")


class AlpacaTemplate(PromptTemplate):
    template_input = (
        "Below is an instruction that describes a task, "
        "paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:"
        "\n{context}\n\n### Response:\n"
    )
    template_no_input = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n"
    )
    soft_prompt = (
        "Generate a clear and direct response that accurately fulfills "
        "the task based on the provided instruction and any given context."
    )

    def get_prompt(self, batch):
        sources = []
        targets = []
        for instruction, context, response in zip(batch["instruction"], batch["context"], batch["response"]):
            example = {"instruction": instruction, "context": context}
            sources.append(
                self.template_input.format_map(example) if example.get("context", "") != ""
                else self.template_no_input.format_map(example)
            )
            targets.append(response + self.tokenizer.eos_token)
        return sources, targets


class SQuADTemplate(PromptTemplate):
    template = "Title: {title}\nContext: {context}\nQuestion: {question}\nAnswer: "

    def get_prompt(self, batch):
        sources = []
        targets = []
        for title, context, question, answers in zip(
            batch["title"], batch["context"], batch["question"], batch["answers"]
        ):
            example = {"title": title, "context": context, "question": question}
            sources.append(self.template.format_map(example))
            targets.append(answers["text"][0] + self.tokenizer.eos_token)
        return sources, targets


class MSMarcoTemplate(PromptTemplate):
    template = "Passage: {passage}\nQuestion: {query}\nAnswer: "

    def get_prompt(self, batch):
        sources = []
        targets = []
        for passages, query, answers in zip(
            batch["passages"], batch["query"], batch["answers"]
        ):
            example = {"passage": passages["passage_text"][0], "query": query}
            sources.append(self.template.format_map(example))
            targets.append(answers[0] + self.tokenizer.eos_token)
        return sources, targets


class BoolQTemplate(PromptTemplate):
    template = "{passage} {question} "
    choices = ["No", "Yes"]
    soft_prompt = (
        "Based on the passage, answer 'Yes' if the question is supported, otherwise answer 'No'."
    )

    def get_prompt(self, batch) -> Tuple[List[str], List[str]]:
        sources = []
        for passage, question in zip(batch["passage"], batch["question"]):
            if not question.endswith("?"):
                question = question + "?"
            question = question[0].upper() + question[1:]
            example = {"passage": passage, "question": question}
            sources.append(self.template.format_map(example))
        if self.train:
            targets = [self.choices[choice_idx] + self.tokenizer.eos_token for choice_idx in batch['class']]
            return sources, targets
        else:
            choices = [choice + self.tokenizer.eos_token for choice in self.choices]
            return sources, choices


class RACETemplate(PromptTemplate):
    template = "Article: {article}\n\nQuestion: {question}\n{options}\n\nAnswer: "
    choices = ["A", "B", "C", "D"]

    def get_prompt(self, batch) -> Tuple[List[str], List[str]]:
        sources = []
        for article, question, options in zip(batch["article"], batch["question"], batch["options"]):
            options = "\n".join([f"{choice}) {option}" for choice, option in zip(self.choices, options)])
            example = {"article": article, "question": question, "options": options}
            sources.append(self.template.format_map(example))
        if self.train:
            targets = [self.choices[choice_idx] + self.tokenizer.eos_token for choice_idx in batch['class']]
            return sources, targets
        else:
            choices = [choice + self.tokenizer.eos_token for choice in self.choices]
            return sources, choices


class WiCTemplate(PromptTemplate):
    template = (
        "Does the word \"{word}\" have the same meaning in these two sentences? "
        "Yes, No?\n{sentence1}\n{sentence2}\n"
    )
    choices = ["No", "Yes"]
    soft_prompt = (
        "Determine if the word has the same meaning in both sentences based on context."
    )

    def get_prompt(self, batch) -> Tuple[List[str], List[str]]:
        sources = []
        for word, sentence1, sentence2 in zip(batch["word"], batch["sentence1"], batch["sentence2"]):
            example = {"word": word, "sentence1": sentence1, "sentence2": sentence2}
            sources.append(self.template.format_map(example))
        if self.train:
            targets = [self.choices[choice_idx] + self.tokenizer.eos_token for choice_idx in batch['class']]
            return sources, targets
        else:
            choices = [choice + self.tokenizer.eos_token for choice in self.choices]
            return sources, choices


class RTETemplate(PromptTemplate):
    template = "{premise}\nDoes this mean that \"{hypothesis}\" is true? Yes or No? "
    choices = ["Yes", "No"]

    def get_prompt(self, batch) -> Tuple[List[str], List[str]]:
        sources = []
        for premise, hypothesis in zip(batch["premise"], batch["hypothesis"]):
            example = {"premise": premise, "hypothesis": hypothesis}
            sources.append(self.template.format_map(example))
        if self.train:
            targets = [self.choices[choice_idx] + self.tokenizer.eos_token for choice_idx in batch['class']]
            return sources, targets
        else:
            choices = [choice + self.tokenizer.eos_token for choice in self.choices]
            return sources, choices


class SNLITemplate(PromptTemplate):
    template = "Suppose {premise} Can we infer that \"{hypothesis}\"? Yes, No, or Maybe?\n"
    choices = ["Yes", "Maybe", "No"]
    soft_prompt = (
        "Based on the premise, determine if the hypothesis logically follows. "
        "Answer with ‘Yes’ if it does, ‘No’ if it doesn’t, or ‘Maybe’ if uncertain."
    )

    def get_prompt(self, batch) -> Tuple[List[str], List[str]]:
        sources = []
        for premise, hypothesis in zip(batch["premise"], batch["hypothesis"]):
            example = {"premise": premise, "hypothesis": hypothesis}
            sources.append(self.template.format_map(example))
        if self.train:
            targets = [self.choices[choice_idx] + self.tokenizer.eos_token for choice_idx in batch['class']]
            return sources, targets
        else:
            choices = [choice + self.tokenizer.eos_token for choice in self.choices]
            return sources, choices


class SST5Template(PromptTemplate):
    template = "Classify the sentiment of the following sentence into one of five categories: " \
               "Very Negative, Negative, Neutral, Positive, or Very Positive.\n"\
               "Sentence: {text}\nSentiment: "
    choices = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]
    soft_prompt = (
        "Evaluate sentiment intensity and emotional tone."
    )

    def get_prompt(self, batch) -> Tuple[List[str], List[str]]:
        sources = []
        for text in batch["text"]:
            sources.append(self.template.format(text=text))
        if self.train:
            targets = [self.choices[choice_idx] + self.tokenizer.eos_token for choice_idx in batch['class']]
            return sources, targets
        else:
            choices = [choice + self.tokenizer.eos_token for choice in self.choices]
            return sources, choices


class AGNewsTemplate(PromptTemplate):
    template = "What is the most accurate label for this news article?\n{text}\n"
    choices = ["World", "Sports", "Business", "Sci/Tech"]
    soft_prompt = (
        "You are a knowledgeable assistant trained to categorize news articles into specific domains."
    )

    def get_prompt(self, batch) -> Tuple[List[str], List[str]]:
        sources = [self.template.format(text=text) for text in batch["text"]]
        if self.train:
            targets = [self.choices[choice_idx] + self.tokenizer.eos_token for choice_idx in batch['class']]
            return sources, targets
        else:
            choices = [choice + self.tokenizer.eos_token for choice in self.choices]
            return sources, choices


class MasakhaNewsTemplate(PromptTemplate):
    template = "What is the most accurate label for this news article?\n{text}\n"
    choices = ["business", "politics", "sports", "technology"]

    def get_prompt(self, batch) -> Tuple[List[str], List[str]]:
        sources = [self.template.format(text=text) for text in batch["text"]]
        if self.train:
            targets = [self.choices[choice_idx] + self.tokenizer.eos_token for choice_idx in batch['class']]
            return sources, targets
        else:
            choices = [choice + self.tokenizer.eos_token for choice in self.choices]
            return sources, choices


class YelpTemplate(PromptTemplate):
    template = "Based on the following review text, determine the star rating (from 1 to 5):\n" \
               "Review Text: {text}\nStar Rating (1-5): "
    choices = ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]
    soft_prompt = (
        "Assign a star rating based on the review's overall sentiment. "
        "1-2 stars for negative feedback, 3 stars for neutral, and 4-5 stars for positive feedback."
    )

    def get_prompt(self, batch) -> Tuple[List[str], List[str]]:
        sources = [self.template.format(text=text) for text in batch["text"]]
        if self.train:
            targets = [self.choices[choice_idx] + self.tokenizer.eos_token for choice_idx in batch['class']]
            return sources, targets
        else:
            choices = [choice + self.tokenizer.eos_token for choice in self.choices]
            return sources, choices


class YelpPolarityTemplate(PromptTemplate):
    template = "Based on the following review text, determine the sentiment polarity (positive or negative):\n" \
               "Review Text: {text}\nSentiment Polarity (positive/negative): "
    choices = ["negative", "positive"]
    soft_prompt = (
        "Identify the polarity of the review, where positive indicates a favorable sentiment "
        "and negative indicates an unfavorable sentiment."
    )

    def get_prompt(self, batch) -> Tuple[List[str], List[str]]:
        sources = [self.template.format(text=text) for text in batch["text"]]
        if self.train:
            targets = [self.choices[choice_idx] + self.tokenizer.eos_token for choice_idx in batch['class']]
            return sources, targets
        else:
            choices = [choice + self.tokenizer.eos_token for choice in self.choices]
            return sources, choices


DATASET_TO_TEMPLATE = {
    "dolly-15k": AlpacaTemplate,
    "alpaca": AlpacaTemplate,
    "natural-instruct": AlpacaTemplate,
    "squad": SQuADTemplate,
    "ms-marco": MSMarcoTemplate,
    "boolq": BoolQTemplate,
    "race": RACETemplate,
    "wic": WiCTemplate,
    "mcl-wic": WiCTemplate,
    "rte": RTETemplate,
    "snli": SNLITemplate,
    "mnli": SNLITemplate,
    "sst-5": SST5Template,
    "ag-news": AGNewsTemplate,
    "masakha-news": MasakhaNewsTemplate,
    "yelp": YelpTemplate,
    "yelp-p": YelpPolarityTemplate
}
