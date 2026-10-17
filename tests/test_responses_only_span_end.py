# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Where ``train_on_responses_only`` ends an assistant span.

An unterminated last turn must be kept to the end of the row, and a model's stop ids
(GLM ``<|observation|>`` / ``<|user|>``, Phi-4-mini ``<|end|>``), which are not
``tokenizer.eos_token``, must end a span inclusively. CPU-pure stub tokenizers; the
real-tokenizer case skips offline.
"""

from types import SimpleNamespace

import pytest

from unsloth_zoo.dataset_utils import train_on_responses_only


class _AddedToken:
    def __init__(self, content, special):
        self.content = content
        self.special = special


class CharTokenizer:
    """One id per character, no specials."""
    eos_token_id = None
    bos_token_id = None
    added_tokens_decoder = {}
    all_special_ids = []

    def __call__(self, text, add_special_tokens = False):
        return SimpleNamespace(input_ids = [ord(c) for c in text])


def _char_trained(text):
    fn = train_on_responses_only(
        None, "\n### User:", "\n### Asst:", force_match = False,
        tokenizer = CharTokenizer(), return_function = True,
    )
    ids = [ord(c) for c in text]
    labels = fn({"input_ids": [ids]})["labels"][0]
    return "".join(chr(l) for l in labels if l != -100)


@pytest.mark.parametrize("tail", ["line1\nZ", "line1Z", "line1\n"])
def test_unterminated_last_turn_is_kept_to_the_end(tail):
    assert _char_trained("\n### User:hi\n### Asst:" + tail) == tail


def test_user_marker_still_ends_the_span():
    assert _char_trained("\n### User:hi\n### Asst:A1\n### User:q\n### Asst:A2") == "A1A2"


# GLM shape: role tokens share no opener, eos is <|endoftext|>, the turn stops are
# <|observation|> (after a tool call) and <|user|> (after an answer).
SYSTEM, USER, ASSISTANT, OBSERVATION, EOS, NL = 1, 2, 3, 4, 5, 6
QUESTION, TOOLCALL, TOOLRESULT, FINAL = 101, 102, 103, 104
GLM_VOCAB = {"<|system|>": SYSTEM, "<|user|>": USER, "<|assistant|>": ASSISTANT,
             "<|observation|>": OBSERVATION, "<|endoftext|>": EOS, "\n": NL}


class GLMTokenizer:
    def __init__(self):
        names = {i: t for t, i in GLM_VOCAB.items()}
        self.added_tokens_decoder = {i: _AddedToken(names[i], True) for i in (SYSTEM, USER, ASSISTANT, OBSERVATION, EOS)}
        self.all_special_ids = [EOS]
        self.bos_token_id = None
        self.eos_token_id = EOS

    def __call__(self, text, add_special_tokens = False):
        ids, rest = [], text
        while rest:
            token = next(t for t in GLM_VOCAB if rest.startswith(t))
            ids.append(GLM_VOCAB[token])
            rest = rest[len(token):]
        return SimpleNamespace(input_ids = ids)


GLM_ROW = [USER, NL, QUESTION, ASSISTANT, NL, TOOLCALL, OBSERVATION, NL, TOOLRESULT,
           ASSISTANT, NL, FINAL, USER, NL, QUESTION, ASSISTANT, NL, FINAL]


def _glm_trained(**kwargs):
    kwargs.setdefault("tokenizer", GLMTokenizer())
    fn = train_on_responses_only(
        kwargs.pop("trainer", None), "<|user|>\n", "<|assistant|>\n", return_function = True, **kwargs,
    )
    labels = fn({"input_ids": [GLM_ROW]})["labels"][0]
    return [l for l in labels if l != -100]


def test_stop_token_ids_end_a_span_inclusively():
    trained = _glm_trained(stop_token_ids = [USER, OBSERVATION, EOS])
    assert TOOLRESULT not in trained, "the tool result was trained on"
    assert trained == [TOOLCALL, OBSERVATION, FINAL, USER, FINAL]


def test_stop_ids_come_from_the_trainer_generation_config():
    generation_config = SimpleNamespace(eos_token_id = [EOS, USER, OBSERVATION])
    trainer = SimpleNamespace(model = SimpleNamespace(generation_config = generation_config))
    trained = _glm_trained(trainer = trainer)
    assert TOOLRESULT not in trained
    assert OBSERVATION in trained and USER in trained


def test_a_stop_id_equal_to_the_response_marker_is_ignored():
    trained = _glm_trained(stop_token_ids = [ASSISTANT, OBSERVATION])
    assert trained == [TOOLCALL, OBSERVATION, FINAL, FINAL]


TOOLS = [{"type": "function", "function": {
    "name": "multiply", "description": "Multiply two integers.",
    "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]},
}}]
CONVERSATION = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "What is 77 * 88? USERQ_1"},
    {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "multiply", "arguments": {"a": 77, "b": 88}}}]},
    {"role": "tool", "name": "multiply", "content": "TOOLOUT_9137"},
    {"role": "assistant", "content": "FINALANS_5521"},
]


@pytest.mark.parametrize("repo", ["unsloth/GLM-4.7-Flash", "microsoft/Phi-4-mini-instruct"])
def test_real_templates_mask_tool_results_with_generation_stop_ids(repo):
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(repo)
        generation_config = transformers.GenerationConfig.from_pretrained(repo)
    except Exception as e:  # offline runner
        pytest.skip(f"cannot load {repo}: {e}")
    text = tokenizer.apply_chat_template(CONVERSATION, tools = TOOLS, tokenize = False)
    input_ids = tokenizer(text, add_special_tokens = False)["input_ids"]
    trainer = SimpleNamespace(model = SimpleNamespace(generation_config = generation_config))
    labels = train_on_responses_only(trainer, tokenizer = tokenizer, return_function = True)(
        {"input_ids": [input_ids]}
    )["labels"][0]
    trained = tokenizer.decode([t for t, l in zip(input_ids, labels) if l != -100])
    assert "TOOLOUT" not in trained and "USERQ" not in trained, trained
    assert '"a": 77' in trained or '"a":77' in trained, trained
    assert "FINALANS_5521" in trained, trained
