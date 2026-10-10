"""Loads vocab.toml, the single source of truth for tokens."""
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Token:
    id: int
    name: str
    group: str
    args: tuple = ()
    ret: str = ""
    vsa: str = "atom"
    note: str = ""
    family: str = ""  # "R", "V" or "NEW" for numbered ranges
    index: int = -1   # position inside a numbered range

    @property
    def arity(self):
        return len(self.args)


@dataclass
class Vocab:
    groups: dict = field(default_factory=dict)   # name -> what it covers, in bootstrap order
    tokens: list = field(default_factory=list)
    by_name: dict = field(default_factory=dict)
    by_id: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path=Path(__file__).with_name("vocab.toml")):
        v = cls()
        seen = set()
        table = tomllib.loads(Path(path).read_text())
        v.groups = {g["name"]: g["about"] for g in table["group"]}
        for row in table["token"]:
            if row["group"] not in v.groups:
                raise ValueError(f'{row["name"]}: unknown group {row["group"]!r}')
            count = row.get("count")
            names = [f'{row["name"]}{i}' for i in range(count)] if count else [row["name"]]
            for i, name in enumerate(names):
                tid = row["id"] + i
                if tid in seen:
                    raise ValueError(f"duplicate token id {tid} ({name})")
                seen.add(tid)
                tok = Token(id=tid, name=name, group=row["group"], args=tuple(row.get("args", ())),
                            ret=row.get("ret", ""), vsa=row.get("vsa", "atom"), note=row.get("note", ""),
                            family=row["name"] if count else "", index=i if count else -1)
                v.tokens.append(tok)
                v.by_name[name] = tok
                v.by_id[tid] = tok
        return v

    def __getitem__(self, name):
        return self.by_name[name]

    def __contains__(self, name):
        return name in self.by_name


VOCAB = Vocab.load()
