# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""DeepSeek-V4.1's serve path without an engine: its chat template against the golden prompts, the DSML tool calls
(the translator and the parser behind it), the forced opening, the template's call form, and over both APIs against a
fake engine.  The template, translator, parser and template-choice cases are upstream Strata's (serve/test_dsv41.py).

    python -m unittest serve.test_dsml -v
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from serve.dsml import DsmlOutputParser, DsmlTranslator, to_dsml_force  # noqa: E402
from serve.frontend import ChatTemplate, OutputParser, forced_call, openai_to_messages  # noqa: E402

FIX = ROOT / "serve" / "fixtures" / "deepseek41"
TEMPLATE = FIX / "chat_template.jinja"
GOLDEN = json.loads((FIX / "golden.json").read_text(encoding="utf-8"))


def call(*invokes, string="true") -> str:
    """The text DeepSeek writes for calls: [(name, [(param, value, is_string)...])...]."""
    out = ["\n\n<｜DSML｜ calls>"]
    for name, params in invokes:
        out.append(f'<｜DSML｜ invoke name="{name}">')
        for p, v, is_str in params:
            out.append(f'<｜DSML｜ parameter name="{p}" string="{"true" if is_str else "false"}">{v}</｜DSML｜ parameter>')
        out.append("</｜DSML｜ invoke>")
    out.append("</｜DSML｜ calls>")
    return "\n".join(out)


WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "weather",
                                              "parameters": {"type": "object", "properties": {
                                                  "city": {"type": "string"}, "days": {"type": "integer"}},
                                                  "required": ["city"]}}}


class Template(unittest.TestCase):
    """The GGUF's own chat template (the same bytes as DeepSeek's encoding.py on the cases checked), through the
    server's request reader and renderer: the text llama.cpp tokenized for the reference run."""

    @classmethod
    def setUpClass(cls):
        cls.tpl = ChatTemplate(TEMPLATE)

    def test_the_golden_prompts(self):
        for case in GOLDEN:
            messages, tools, kw = openai_to_messages(case["request"])
            self.assertEqual(self.tpl.render(messages, tools=tools, **kw), case["text"], case["id"] + " " + case["mode"])

    def test_thinking_is_the_default(self):
        text = self.tpl.render([{"role": "user", "content": "hi"}])
        self.assertTrue(text.endswith("<｜Assistant｜><think>"))
        text = self.tpl.render([{"role": "user", "content": "hi"}], enable_thinking=False)
        self.assertTrue(text.endswith("<｜Assistant｜></think>"))

    def test_a_tool_call_and_its_result_render_in_dsml(self):
        msgs = [{"role": "user", "content": "weather?"},
                {"role": "assistant", "content": "", "reasoning_content": "call it",
                 "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Zagreb", "days": 3}}}]},
                {"role": "tool", "content": "sunny"}]
        text = self.tpl.render(msgs, tools=[WEATHER["function"]], enable_thinking=False)
        self.assertIn('<｜DSML｜ invoke name="get_weather">', text)
        self.assertIn('<｜DSML｜ parameter name="city" string="true">Zagreb</｜DSML｜ parameter>', text)
        self.assertIn('<｜DSML｜ parameter name="days" string="false">3</｜DSML｜ parameter>', text)
        self.assertIn("<tool_result>sunny</tool_result>", text)


class Translator(unittest.TestCase):
    RAW = call(("get_weather", [("city", "Zagreb", True), ("days", "3", False)]), ("ping", []))

    def all_at_once(self, text):
        t = DsmlTranslator()
        return t.feed(text) + t.flush()

    def test_a_call_becomes_the_qwen_form(self):
        got = self.all_at_once("Sure." + self.RAW)
        self.assertEqual(got, "Sure.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nZagreb\n</parameter>\n"
                              "<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>\n"
                              "<tool_call>\n<function=ping>\n</function>\n</tool_call>\n")

    def test_the_split_of_the_stream_changes_nothing(self):
        want = self.all_at_once("a < b and 1<2" + self.RAW + "tail")
        for step in (1, 2, 3, 5, 8, 13):
            t = DsmlTranslator()
            text = "a < b and 1<2" + self.RAW + "tail"
            got = "".join(t.feed(text[i:i + step]) for i in range(0, len(text), step)) + t.flush()
            self.assertEqual(got, want, step)

    def test_text_with_angle_brackets_passes(self):
        text = "x <div> y </｜DSML｜ nope> <｜User｜> <｜DSML｜ unknown>"
        self.assertEqual(self.all_at_once(text), text)

    def test_the_end_of_the_output_inside_a_tag_is_text(self):
        t = DsmlTranslator()
        got = t.feed("ok <｜DSML｜ inv") + t.flush()
        self.assertEqual(got, "ok <｜DSML｜ inv")

    def test_a_value_that_looks_like_a_tag(self):
        raw = call(("write", [("text", "<b>bold</b> and < 3", True)]))
        self.assertIn("<parameter=text>\n<b>bold</b> and < 3\n</parameter>", self.all_at_once(raw))

    def test_a_missing_closing_tag_is_closed(self):
        raw = '<｜DSML｜ calls>\n<｜DSML｜ invoke name="f">\n<｜DSML｜ parameter name="a" string="true">x</｜DSML｜ invoke>'
        self.assertEqual(self.all_at_once(raw),
                         "<tool_call>\n<function=f>\n<parameter=a>\nx\n</parameter>\n</function>\n</tool_call>\n")


class Parser(unittest.TestCase):
    TOOLS = [WEATHER["function"], {"name": "lookup", "description": "", "parameters": {
        "type": "object", "properties": {"id": {}}}}]

    def run_parser(self, text, thinking=True, step=7, cls=DsmlOutputParser):
        p = cls(thinking=thinking, tools=self.TOOLS, stream_tools=True)
        evs = []
        for i in range(0, len(text), step):
            evs += p.feed(text[i:i + step])
        evs += p.finish("stop")
        return evs

    def test_reasoning_then_two_calls(self):
        raw = call(("get_weather", [("city", "Zagreb", True), ("days", "3", False)]), ("lookup", [("id", "123", True)]))
        for step in (1, 4, 1000):
            evs = self.run_parser("Think < ok.</think>" + raw, step=step)
            self.assertEqual("".join(e.text for e in evs if e.kind == "reasoning"), "Think < ok.")
            calls = [e.call for e in evs if e.kind == "tool_call"]
            self.assertEqual([(c.name, c.arguments) for c in calls],
                             [("get_weather", {"city": "Zagreb", "days": 3}), ("lookup", {"id": "123"})])
            self.assertEqual("".join(e.text for e in evs if e.kind == "tool_args"),
                             '{"city":"Zagreb","days":3}{"id":"123"}')

    def test_a_string_value_stays_a_string_even_when_the_schema_says_nothing(self):
        evs = self.run_parser(call(("lookup", [("id", "007", True)])), thinking=False)
        self.assertEqual([e.call.arguments for e in evs if e.kind == "tool_call"], [{"id": "007"}])
        evs = self.run_parser(call(("lookup", [("id", "7", False)])), thinking=False)
        self.assertEqual([e.call.arguments for e in evs if e.kind == "tool_call"], [{"id": 7}])

    def test_a_non_string_value_is_json(self):
        evs = self.run_parser(call(("lookup", [("id", '{"a": [1, 2]}', False)])), thinking=False)
        self.assertEqual([e.call.arguments for e in evs if e.kind == "tool_call"], [{"id": {"a": [1, 2]}}])

    def test_content_before_the_call_and_none_after(self):
        evs = self.run_parser("Checking.\n\n" + call(("lookup", [("id", "x", True)]))[2:], thinking=False)
        self.assertEqual("".join(e.text for e in evs if e.kind == "content").strip(), "Checking.")
        self.assertEqual(sum(e.kind == "tool_call" for e in evs), 1)

    def test_a_call_cut_off_by_the_length_limit_is_not_a_call(self):
        raw = call(("get_weather", [("city", "Zagr", True)]))
        raw = raw[:raw.index("Zagr") + 4]
        p = DsmlOutputParser(thinking=False, tools=self.TOOLS, stream_tools=True)
        evs = p.feed(raw) + p.finish("length")
        self.assertEqual(sum(e.kind == "tool_call" for e in evs), 0)

    def test_the_plain_parser_is_unchanged(self):
        p = OutputParser(thinking=False, tools=self.TOOLS)
        evs = p.feed("<tool_call>\n<function=lookup>\n<parameter=id>\n5\n</parameter>\n</function>\n</tool_call>") + p.finish("stop")   # #4701: the end of the turn settles a closed call
        self.assertEqual([e.call.arguments for e in evs if e.kind == "tool_call"], [{"id": 5}])

    def test_clean_counts_a_held_tag(self):
        p = DsmlOutputParser(thinking=False, tools=self.TOOLS)
        p.feed("hello <｜DS")
        self.assertFalse(p.clean)
        p.feed("ML｜ calls>")
        self.assertTrue(p.clean)

    def test_forced_opening(self):
        self.assertEqual(to_dsml_force("<tool_call>\n<function=get_weather>\n"),
                         '<｜DSML｜ calls>\n<｜DSML｜ invoke name="get_weather">\n')
        self.assertEqual(to_dsml_force("<tool_call>\n<function="), '<｜DSML｜ calls>\n<｜DSML｜ invoke name="')
        self.assertIsNone(to_dsml_force(None))
        # the opening and the model's continuation are one call
        p = DsmlOutputParser(thinking=False, tools=self.TOOLS, stream_tools=True)
        evs = p.feed(to_dsml_force("<tool_call>\n<function=get_weather>\n"))
        evs += p.feed('<｜DSML｜ parameter name="city" string="true">Rome</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>')
        evs += p.finish("stop")
        self.assertEqual([(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"],
                         [("get_weather", {"city": "Rome"})])


class PickTemplateTests(unittest.TestCase):
    """A pack made from another GGUF can ship the V4-Flash template (`tool_calls`, no space): the V4.1 model then
    writes malformed calls (P2003 install test).  The server swaps in the V4.1 template for such a pack."""

    def test_v4flash_pack_template_is_replaced(self):
        import tempfile
        from serve import dsml
        with tempfile.TemporaryDirectory() as d:
            old = Path(d) / "old.jinja"
            old.write_text("{%- set dsml_token = '｜DSML｜' -%}{{ '<' + dsml_token + 'tool_calls>' }}", encoding="utf-8")
            self.assertEqual(dsml.pick_template(old), (dsml.FIXTURE_TEMPLATE, True))
            same = Path(d) / "same.jinja"
            same.write_text(dsml.FIXTURE_TEMPLATE.read_text(encoding="utf-8"), encoding="utf-8")
            self.assertEqual(dsml.pick_template(same), (same, False))
            self.assertEqual(dsml.pick_template(Path(d) / "none.jinja"), (dsml.FIXTURE_TEMPLATE, False))

    def test_fixture_is_accepted(self):
        from serve import dsml
        self.assertTrue(dsml.v41_template_ok(dsml.FIXTURE_TEMPLATE.read_text(encoding="utf-8")))


class CallForm(unittest.TestCase):
    """XeMaya: the server picks the reader and the forced opening from the template's call form."""

    def test_the_deepseek_template_is_the_dsml_form_and_the_others_are_not(self):
        self.assertEqual(ChatTemplate(TEMPLATE).call_form, "dsml")
        self.assertEqual(ChatTemplate(ROOT / "serve" / "test_chat_template.jinja").call_form, "qwen")

    def test_a_forced_call_opens_in_dsml(self):
        tools = [WEATHER["function"], {"name": "lookup", "parameters": {"type": "object", "properties": {}}}]
        named = {"type": "function", "function": {"name": "get_weather"}}
        self.assertEqual(forced_call(named, tools, "dsml"), '<｜DSML｜ calls>\n<｜DSML｜ invoke name="get_weather">\n')
        self.assertEqual(forced_call("required", tools, "dsml"), '<｜DSML｜ calls>\n<｜DSML｜ invoke name="')
        self.assertIsNone(forced_call("auto", tools, "dsml"))
        with self.assertRaises(ValueError):
            forced_call({"type": "function", "function": {"name": "nope"}}, tools, "dsml")
        # the other forms are as they were
        self.assertEqual(forced_call(named, tools), "<tool_call>\n<function=get_weather>\n")

    def test_no_closing_think_tag_makes_the_text_the_answer(self):
        p = DsmlOutputParser(thinking=True, tools=[])
        evs = p.feed("The answer is 4.") + p.finish("stop")
        self.assertEqual([e.text for e in evs if e.kind == "content"], ["The answer is 4."])
        # ... but a turn cut by the token limit is not turned into an answer
        p = DsmlOutputParser(thinking=True, tools=[])
        evs = p.feed("Still thinking") + p.finish("length")
        self.assertEqual([e.text for e in evs if e.kind == "content"], [])


class OverHttp(unittest.TestCase):
    """The whole serve path against the fake engine: the DeepSeek template in, the model's DSML out, both APIs."""

    SCRIPT = "The user wants the weather.</think>" + call(("get_weather", [("city", "Zagreb", True), ("days", "3", False)]))

    def ask(self, path, body, headers=None):
        import urllib.request
        from serve.server import ByteTokenizer, MockEngine, Service, serve
        tok = ByteTokenizer()
        engine = MockEngine(tok, self.SCRIPT, max_context=8192)
        httpd = serve(Service(engine, tok, ChatTemplate(TEMPLATE)), port=0)
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_address[1]}{path}", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json", **(headers or {})})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read().decode()
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_openai_tool_call(self):
        body = {"model": "x", "max_tokens": 400, "messages": [{"role": "user", "content": "weather in Zagreb?"}], "tools": [WEATHER]}
        msg = json.loads(self.ask("/v1/chat/completions", body))["choices"][0]
        self.assertEqual(msg["finish_reason"], "tool_calls")
        fn = msg["message"]["tool_calls"][0]["function"]
        self.assertEqual((fn["name"], json.loads(fn["arguments"])), ("get_weather", {"city": "Zagreb", "days": 3}))
        self.assertEqual(msg["message"]["reasoning_content"], "The user wants the weather.")
        self.assertNotIn("DSML", json.dumps(msg, ensure_ascii=False))

    def test_openai_streamed_tool_call(self):
        body = {"model": "x", "max_tokens": 400, "stream": True, "messages": [{"role": "user", "content": "weather?"}], "tools": [WEATHER]}
        chunks = [json.loads(line[6:]) for line in self.ask("/v1/chat/completions", body).splitlines() if line.startswith("data: {")]
        deltas = [c["choices"][0]["delta"] for c in chunks if c.get("choices")]
        args = "".join(t["function"].get("arguments") or "" for d in deltas for t in d.get("tool_calls") or [])
        self.assertEqual(json.loads(args), {"city": "Zagreb", "days": 3})
        self.assertNotIn("DSML", "".join(d.get("content") or "" for d in deltas))

    def test_anthropic_tool_use(self):
        body = {"model": "x", "max_tokens": 400, "messages": [{"role": "user", "content": "weather?"}],
                "tools": [{"name": "get_weather", "input_schema": WEATHER["function"]["parameters"]}]}
        msg = json.loads(self.ask("/v1/messages", body, {"anthropic-version": "2023-06-01"}))
        uses = [b for b in msg["content"] if b["type"] == "tool_use"]
        self.assertEqual((msg["stop_reason"], uses[0]["name"], uses[0]["input"]), ("tool_use", "get_weather", {"city": "Zagreb", "days": 3}))


if __name__ == "__main__":
    unittest.main()
