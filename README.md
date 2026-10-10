# primer

A small typed language for training from-scratch models: a vocabulary of a few hundred tokens built up in layers (syntax, logic, values, computation, teaching frames, episodes), each explainable using only the layers before it, plus an interpreter that checks every claim made in training data.

- [SPEC.md](SPEC.md): the language
- [primer/vocab.toml](primer/vocab.toml): every token, the single source of truth
- [docs/vocab-v0.md](docs/vocab-v0.md) and [docs/design-notes.md](docs/design-notes.md): where the design came from
- [tests/examples.md](tests/examples.md): hand-written example training data, all checked by the tests

```
python3 -m unittest discover -s tests -t .
python3 -c "from primer.check import check; print(check('BOS ADD INT<2> INT<3> INT<5> END'))"
python3 -m primer.gen --groups 1-6 --count 100000 --out data/   # checked training data, one process per core
```

Needs Python 3.11+ and numpy.
