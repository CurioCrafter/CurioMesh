# Contributing to CurioMesh

Questions, ideas and show-and-tell go in the [Discussions](https://github.com/CurioCrafter/CurioMesh/discussions).
Bugs go in an [issue](https://github.com/CurioCrafter/CurioMesh/issues/new) (the add-on's **Report a Problem** button
fills in your versions).

## Bad spots and test cases

A spot CurioMesh got wrong, or a better version of it, is the most useful thing you can send.

1. Mark the bad parts and press **Save for a Report** in the panel (agents: `parts.report`). It writes a zip with the
   part files, `report.md` and `settings.json`. See [docs/PARTS_EXCHANGE.md](docs/PARTS_EXCHANGE.md).
2. Fill in the [Bad spot or fixed part](https://github.com/CurioCrafter/CurioMesh/issues/new?template=bad-spot.yml)
   form and drag the zip into it.

CurioMesh never uploads anything. The zip holds pieces of your mesh, so send only meshes you have the right to share.
The form asks you to agree that what you send may be used as a CurioMesh test case under
[CC0](https://creativecommons.org/publicdomain/zero/1.0/).

## Code

Pull requests are welcome. Run the tests first:

```powershell
python -m unittest discover -s tests -v
ruff check .
```

Write your own code. Do not copy code, UI, names or workflows from other tools; published papers are fine, with a
citation and a row in [docs/ORIGINALITY.md](docs/ORIGINALITY.md).
