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

"""Mistral-family templates put no ``[/INST]`` before the answer to a tool result:
``[TOOL_RESULTS]...[/TOOL_RESULTS]answer</s>``. ``train_on_responses_only`` must still
train that answer, through a second response marker (explicit list, or auto-detected).
"""

import copy
from types import SimpleNamespace

import pytest

from unsloth_zoo.dataset_utils import train_on_responses_only


class _AddedToken:
    def __init__(self, content, special):
        self.content = content
        self.special = special


BOS, EOS, INST, END_INST, TOOL_CALLS, TOOL_RESULTS, END_TOOL_RESULTS = 1, 2, 3, 4, 5, 6, 7
QUESTION, TOOLCALL, TOOLRESULT, FINAL = 101, 102, 103, 104
VOCAB = {"<s>": BOS, "</s>": EOS, "[INST]": INST, "[/INST]": END_INST, "[TOOL_CALLS]": TOOL_CALLS,
         "[TOOL_RESULTS]": TOOL_RESULTS, "[/TOOL_RESULTS]": END_TOOL_RESULTS}


class MistralStubTokenizer:
    def __init__(self):
        names = {i: t for t, i in VOCAB.items()}
        self.added_tokens_decoder = {i: _AddedToken(names[i], True) for i in names}
        self.all_special_ids = [BOS, EOS]
        self.bos_token_id = BOS
        self.eos_token_id = EOS

    def __call__(self, text, add_special_tokens = False):
        ids, rest = [], text
        while rest:
            token = next(t for t in VOCAB if rest.startswith(t))
            ids.append(VOCAB[token])
            rest = rest[len(token):]
        return SimpleNamespace(input_ids = ids)


ROW = [BOS, INST, QUESTION, END_INST, TOOL_CALLS, TOOLCALL, EOS,
       TOOL_RESULTS, TOOLRESULT, END_TOOL_RESULTS, FINAL, EOS,
       INST, QUESTION, END_INST, FINAL, EOS]


def _trained(**kwargs):
    fn = train_on_responses_only(
        None, "[INST]", ["[/INST]", "[/TOOL_RESULTS]"],
        tokenizer = MistralStubTokenizer(), return_function = True, **kwargs,
    )
    labels = fn({"input_ids": [ROW]})["labels"][0]
    return [l for l in labels if l != -100]


def test_answer_after_tool_result_is_trained():
    assert _trained() == [TOOL_CALLS, TOOLCALL, EOS, FINAL, EOS, FINAL, EOS]


def test_last_response_only_with_a_second_marker():
    assert _trained(last_response_only = True) == [FINAL, EOS]


# Mistral-shaped template over a real cached tokenizer (from the audit's ZX-12-synthetic.py).
MISTRAL_TEMPLATE = (
    "{{ bos_token }}{%- for m in messages -%}"
    "{%- if m.role == 'user' -%}[INST]{{ m.content }}[/INST]"
    "{%- elif m.role == 'system' -%}"
    "{%- elif m.role == 'assistant' -%}{%- if m.tool_calls -%}[TOOL_CALLS]{%- for t in m.tool_calls -%}{{ t.function.name }}[ARGS]{{ t.function.arguments | tojson }}{%- endfor -%}{{ eos_token }}{%- else -%}{{ m.content }}{{ eos_token }}{%- endif -%}"
    "{%- elif m.role == 'tool' -%}[TOOL_RESULTS]{{ m.content }}[/TOOL_RESULTS]{%- endif -%}{%- endfor -%}"
)


def _call(a, b):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"type": "function", "function": {"name": "multiply", "arguments": {"a": a, "b": b}}}]}


CONVERSATION = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "What is 77 * 88? USERQ_1"}, _call(77, 88),
    {"role": "tool", "content": "TOOLOUT_9137"}, {"role": "assistant", "content": "FINALANS_5521"},
    {"role": "user", "content": "And 12 * 13? USERQ_2"}, _call(12, 13),
    {"role": "tool", "content": "TOOLOUT_4242"}, {"role": "assistant", "content": "FINALANS_7777"},
]


def _cached_tokenizer(repo):
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(repo)
    except Exception as e:  # offline runner
        pytest.skip(f"cannot load {repo}: {e}")


def _mistral_shaped():
    tokenizer = copy.deepcopy(_cached_tokenizer("Qwen/Qwen3-0.6B"))
    tokenizer.add_special_tokens({
        "additional_special_tokens": ["[INST]", "[/INST]", "[TOOL_CALLS]", "[ARGS]", "[TOOL_RESULTS]", "[/TOOL_RESULTS]"],
        "eos_token": "</s>", "bos_token": "<s>",
    })
    tokenizer.chat_template = MISTRAL_TEMPLATE
    return tokenizer


def test_auto_detect_trains_the_post_tool_answer():
    from unsloth_zoo.dataset_utils import get_tool_response_part
    tokenizer = _mistral_shaped()
    assert get_tool_response_part(tokenizer, "[INST]", "[/INST]") == "[/TOOL_RESULTS]"
    text = tokenizer.apply_chat_template(CONVERSATION, tokenize = False)
    input_ids = tokenizer(text, add_special_tokens = False)["input_ids"]
    labels = train_on_responses_only(None, tokenizer = tokenizer, return_function = True)(
        {"input_ids": [input_ids]}
    )["labels"][0]
    trained = tokenizer.decode([t for t, l in zip(input_ids, labels) if l != -100])
    assert "FINALANS_5521" in trained and "FINALANS_7777" in trained, trained
    assert '"a": 77' in trained and '"a": 12' in trained, trained
    assert "TOOLOUT" not in trained and "USERQ" not in trained, trained


@pytest.mark.parametrize("repo", ["Qwen/Qwen3-0.6B", "unsloth/Llama-3.2-1B-Instruct"])
def test_templates_with_a_header_after_tools_get_no_second_marker(repo):
    from unsloth_zoo.dataset_utils import get_chat_template_parts, get_tool_response_part
    tokenizer = _cached_tokenizer(repo)
    instruction_part, response_part = get_chat_template_parts(tokenizer)
    assert get_tool_response_part(tokenizer, instruction_part, response_part) is None
