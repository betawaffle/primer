"""Loads vocab.toml, the single source of truth for tokens."""
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

MODEL_ONLY = {"REF", "SLOT", "NEW"}   # appear in the model's id form, never in text


@dataclass(frozen=True)
class Token:
    id: int
    name: str
    group: str
    args: tuple = ()
    ret: str = ""
    vsa: str = "atom"
    note: str = ""
    family: str = ""  # "R", "V" or "NEW" for a name like R3, V0, NEW12
    index: int = -1   # the number in that name

    @property
    def arity(self):
        return len(self.args)


@dataclass
class Vocab:
    groups: dict = field(default_factory=dict)   # name -> what it covers, in bootstrap order
    names: dict = field(default_factory=dict)    # family prefix -> its [[name]] row
    tokens: list = field(default_factory=list)
    by_name: dict = field(default_factory=dict)
    by_id: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path=Path(__file__).with_name("vocab.toml")):
        v = cls()
        seen = set()
        table = tomllib.loads(Path(path).read_text())
        v.groups = {g["name"]: g["about"] for g in table["group"]}
        v.names = {n["prefix"]: n for n in table["name"]}
        for row in table["token"]:
            if row["group"] not in v.groups:
                raise ValueError(f'{row["name"]}: unknown group {row["group"]!r}')
            if row["id"] in seen:
                raise ValueError(f'duplicate token id {row["id"]} ({row["name"]})')
            seen.add(row["id"])
            tok = Token(id=row["id"], name=row["name"], group=row["group"], args=tuple(row.get("args", ())),
                        ret=row.get("ret", ""), vsa=row.get("vsa", "atom"), note=row.get("note", ""))
            v.tokens.append(tok)
            v.by_name[tok.name] = tok
            v.by_id[tok.id] = tok
        v._name_re = re.compile("(" + "|".join(sorted(v.names, key=len, reverse=True)) + r")(\d+)")
        return v

    def name(self, text):
        """The Token for a name such as V0, NEW3 or R12, or None. Names have no
        id: the model sees a binder or a REF in their place."""
        m = self._name_re.fullmatch(text)
        if not m:
            return None
        row = self.names[m.group(1)]
        return Token(id=-1, name=text, group=row["group"], ret="ANY", vsa="ordered", note=row["note"],
                     family=m.group(1), index=int(m.group(2)))

    def __getitem__(self, name):
        tok = self.by_name.get(name) or self.name(name)
        if tok is None:
            raise KeyError(name)
        return tok

    def __contains__(self, name):
        return name in self.by_name or self.name(name) is not None


VOCAB = Vocab.load()
