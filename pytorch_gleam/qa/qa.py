import re
from abc import ABC, abstractmethod
from collections import defaultdict
from string import ascii_lowercase
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn, Tensor
from transformers import AutoTokenizer


class QAModule(nn.Module, ABC):
    def __init__(self):
        r"""
        Question-Answering module for generating question-answering examples and for
        producing question-answering predictions
        """
        super().__init__()

    @abstractmethod
    def generate(
        self, body: str, label: str = None, context: Optional[str] = None
    ) -> Iterable[Tuple[str, Tensor, Tensor, Optional[Tensor]]]:
        r"""
        Create Question-Answering examples from body, context, and QA configuration.

        Args:

            body: Body of QA, in which the answer can be derived.

            label: Optional correct label for example.

            context: Optional QA context, to be provided along with prompt.

        Returns:

            An iterable of tuples, where each tuple is (q_id, input_ids, attention_mask, label)

        """

    @abstractmethod
    def forward(self, qa_ids: List[str], qa_responses: Tensor) -> Tuple[List[str], Tensor]:
        r"""
        Create Question-Answering predictions from qa responses.

        Args:

            qa_ids: List of QA example ids, used to determine which question is being answered

            qa_responses: Tensor of qa response ids to be decoded into answers.

        Returns:

            A tuple of example ids and a Tensor with qa predictions as indices

        """


class MultipleChoiceQAModule(QAModule):
    def __init__(
        self,
        tokenizer_name: str,
        prompt: str,
        label_map: Dict[str, int],
        choice_map: Dict[str, int],
        add_question: bool = False,
    ):
        r"""
        Question-Answering module for generating question-answering examples and for
        producing question-answering predictions

        Args:

            tokenizer_name: Name of tokenizer to use for encoding and decoding.

            prompt: Prompt for QA, question to ask.

            label_map: Map from label name to label idx

            choice_map: Map from choice in choices to label idx from label map

            add_question: Add question mark at end of context and remove period.

        """
        super().__init__()
        self.tokenizer_name = tokenizer_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.prompt = prompt
        self.label_map = label_map
        self.choice_map = choice_map
        self.choices = [k for k, v in self.choice_map.items()]
        self.add_question = add_question

        self.inv_label_map = {v: k for k, v in self.label_map.items()}
        self.inv_choice_map = {v: k for k, v in self.choice_map.items()}
        self.choices_text = " ".join(
            [f"({o_letter}) {o_text}" for o_letter, o_text in zip(ascii_lowercase, self.choices)]
        )

    def generate(
        self, body: str, label: str = None, context: Optional[str] = None
    ) -> Iterable[Tuple[str, Tensor, Tensor, Optional[Tensor]]]:
        r"""
        Create Question-Answering examples from body, context, and QA configuration.

        Args:

            body: Body of QA, in which the answer can be derived.

            label: Optional correct label for example.

            context: Optional QA context, to be provided along with prompt.

        Returns:

            An iterable of tuples, where each tuple is (q_id, input_ids, attention_mask, label)

        """
        # remove twitter urls
        body = re.sub(r"https:\/\/t.co\/[a-zA-Z0-9]{10}", "", body)
        body = body.replace("\t", " ").replace("\n", " ").strip()
        prompt_full = self.prompt
        if context is not None:
            context = context.replace("\t", " ").replace("\n", " ").strip()
            if self.add_question:
                context = context.replace(".", "") + "?"
            prompt_full = f"{self.prompt} {context}"

        qa_text = " \\n ".join([prompt_full, self.choices_text, body])
        qa_text = qa_text.lower()
        token_data = self.tokenizer(qa_text, truncation=True)
        input_ids = token_data["input_ids"]
        attention_mask = token_data["attention_mask"]
        label_ids = None
        if label is not None:
            label_idx = self.label_map[label]
            choice_text = self.inv_choice_map[label_idx]
            choice_text = choice_text.lower()
            label_data = self.tokenizer(choice_text)
            label_ids = label_data["input_ids"]

        q_id = "0"
        yield q_id, input_ids, attention_mask, label_ids

    def forward(self, qa_ids: List[str], qa_responses: Tensor) -> Tuple[List[str], Tensor]:
        r"""
        Create Question-Answering predictions from qa responses.

        Args:

            qa_ids: List of QA example ids, used to determine which question is being answered

            qa_responses: Tensor of qa response ids to be decoded into answers.

        Returns:

            A tuple of example ids and a Tensor with qa predictions as indices

        """
        # List[str]
        qa_response_texts = self.tokenizer.batch_decode(qa_responses, skip_special_tokens=True)
        ex_ids = []
        preds = []
        for qa_id, qa_response in zip(qa_ids, qa_response_texts):
            ex_id, q_id = qa_id.split("||")
            qa_response = qa_response.title()
            if qa_response not in self.choice_map:
                qa_response = self.choices[-1]
            qa_pred = self.choice_map[qa_response]
            ex_ids.append(ex_id)
            preds.append(qa_pred)
        preds = torch.tensor(preds, dtype=torch.long)
        return ex_ids, preds


class MultiTurnQAModule(QAModule):
    def __init__(
        self,
        turns: List[MultipleChoiceQAModule],
        turn_label_map: List[Dict[str, str]],
        label_turn_map: List[Dict[str, str]],
        label_map: Dict[str, int],
    ):
        r"""
        Question-Answering module for generating multi-turn question-answering examples and for
        producing multi-turn question-answering predictions

        Args:

            turns: List of questions to ask

            turn_label_map: Map from turn label name to final label name

            label_turn_map: Map from final label name to turn label name

            label_map: Map from label name to label idx

        """
        super().__init__()
        self.turns = turns
        self.turn_label_map = turn_label_map
        self.label_turn_map = label_turn_map
        self.label_map = label_map

    def generate(
        self, body: str, label: str = None, context: Optional[str] = None
    ) -> Iterable[Tuple[str, Tensor, Tensor, Optional[Tensor]]]:
        r"""
        Create Question-Answering examples from body, context, and QA configuration.

        Args:

            body: Body of QA, in which the answer can be derived.

            label: Optional correct label for example.

            context: Optional QA context, to be provided along with prompt.

        Returns:

            An iterable of tuples, where each tuple is (q_id, input_ids, attention_mask, label)

        """
        for t_idx, (turn, tl_map) in enumerate(zip(self.turns, self.label_turn_map)):
            for t_qid, input_ids, attention_mask, label_ids in turn.generate(body, tl_map[label], context):
                q_id = f"{t_idx}|{t_qid}"
                yield q_id, input_ids, attention_mask, label_ids

    def forward(self, qa_ids: List[str], qa_responses: Tensor) -> Tuple[List[str], Tensor]:
        r"""
        Create Question-Answering predictions from qa responses.

        Args:

            qa_ids: List of QA example ids, used to determine which question is being answered

            qa_responses: Tensor of qa response ids to be decoded into answers.

        Returns:

            A tuple of example ids and a Tensor with qa predictions as indices

        """
        # q_id = f'{t_idx}|{t_qid}'
        turn_ids = defaultdict(list)
        turn_indices = defaultdict(list)
        for ex_idx, qa_id in enumerate(qa_ids):
            ex_id, q_id = qa_id.split("||")
            q_split = q_id.split("|")
            turn_id = q_split[0]
            turn_q_id = "|".join(q_split[1:])
            ex_turn_id = f"{ex_id}||{turn_q_id}"

            turn_idx = int(turn_id)
            turn_ids[turn_idx].append(ex_turn_id)
            turn_indices[turn_idx].append(ex_idx)

        ex_preds_lookup = {}
        for t_idx, turn in enumerate(self.turns):
            t_ids = turn_ids[t_idx]
            t_indices = torch.tensor(turn_indices[t_idx], dtype=torch.long)
            turn_label_map = self.turn_label_map[t_idx]
            t_qa_responses = qa_responses[t_indices]

            ex_turn_ids, ex_turn_preds = turn(t_ids, t_qa_responses)
            for ex_turn_id, ex_turn_pred in zip(ex_turn_ids, ex_turn_preds.tolist()):
                turn_pred_name = turn.inv_label_map[ex_turn_pred]
                pred_name = turn_label_map[turn_pred_name]
                ex_turn_pred = self.label_map[pred_name]
                # positive class - find first turn positive answer
                if ex_turn_id not in ex_preds_lookup:
                    ex_preds_lookup[ex_turn_id] = ex_turn_pred
                elif ex_preds_lookup[ex_turn_id] == 0 and ex_turn_pred > 0:
                    ex_preds_lookup[ex_turn_id] = ex_turn_pred

        ex_ids = []
        ex_preds = []
        for ex_id, ex_pred in ex_preds_lookup.items():
            ex_ids.append(ex_id)
            ex_preds.append(ex_pred)
        ex_preds = torch.tensor(ex_preds, dtype=torch.long)
        return ex_ids, ex_preds
