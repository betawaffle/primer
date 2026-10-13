# primer

A small typed language for training from-scratch models: a vocabulary of a couple of hundred tokens built up in named groups (syntax, logic, values, compute, teach, quote, edit, episode, think, belief, agenda), each explainable using only the groups before it, plus an interpreter that checks every claim made in training data.

- [SPEC.md](SPEC.md): the language
- [primer/vocab.toml](primer/vocab.toml): every token, the single source of truth
- [docs/decisions.md](docs/decisions.md): design decisions made so far that the code doesn't show yet (model, time, context)
- [docs/design-notes.md](docs/design-notes.md): where the vocabulary design came from
- [tests/examples.md](tests/examples.md): hand-written example training data, all checked by the tests
- [primer/english.py](primer/english.py): reads sequences out as pseudocode inside short English sentences, from the `en` templates in vocab.toml (`python3 -m primer.english < data/shard-00000.jsonl`)
- [primer/model.py](primer/model.py): model v0, a small transformer that reads the id form (VSA + learned embeddings, pointer REFs, gated order and time planes); [primer/train.py](primer/train.py) trains it

```
python3 -m unittest discover -s tests -t .
python3 -c "from primer.check import check; print(check('BOS ADD INT<2> INT<3> INT<5> EOS'))"
python3 -m primer.gen --groups syntax-episode --count 100000 --out data/   # checked training data, one process per core
python3 -m primer.gen --split test --count 5000 --out data-test/             # held-out combinations
python3 -m primer.train --train data/ --test data-test/ --steps 20000 --out runs/v0
```

Needs Python 3.11+ and numpy. Training also needs PyTorch (ROCm build on the Framework box; the model tests skip without it).
