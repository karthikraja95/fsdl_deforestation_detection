# FSDL Deforestation Detection

<div align="center">

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/andrecnf/fsdl_deforestation_detection/fsdl_deforestation_detection/dashboard/streamlit_app.py)

[![Dependencies Status](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)](https://github.com/karthikraja95/fsdl_deforestation_detection/pulls?utf8=%E2%9C%93&q=is%3Apr%20author%3Aapp%2Fdependabot)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Security: bandit](https://img.shields.io/badge/security-bandit-green.svg)](https://github.com/PyCQA/bandit)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/karthikraja95/fsdl_deforestation_detection/blob/master/.pre-commit-config.yaml)
[![Semantic Versions](https://img.shields.io/badge/%F0%9F%9A%80-semantic%20versions-informational.svg)](https://github.com/karthikraja95/fsdl_deforestation_detection/releases)
[![License](https://img.shields.io/github/license/karthikraja95/fsdl_deforestation_detection)](https://github.com/karthikraja95/fsdl_deforestation_detection/blob/master/LICENSE)

Detecting deforestation from satellite images: a full stack deep learning project

</div>

## Description

A deep learning approach to detecting deforestation risk, using satellite images and a deep learning model. We relied on [Planet](https://www.planet.com/) imagery from two [Kaggle](https://www.kaggle.com/) datasets (one from the [Amazon rainforest](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space) and another on [oil palm plantations in Borneo](https://www.kaggle.com/c/widsdatathon2019)) and trained a [ResNet](https://paperswithcode.com/method/resnet) model using [FastAI](https://docs.fast.ai/). For more details, check the following links:

* [Streamlit dashboard for testing the model and exploring the data](https://share.streamlit.io/andrecnf/fsdl_deforestation_detection/fsdl_deforestation_detection/dashboard/streamlit_app.py)

* [Model training notebook in Colab](https://colab.research.google.com/github/karthikraja95/fsdl_deforestation_detection/blob/master/fsdl_deforestation_detection/experimental/FSDL_Final_Model.ipynb)

* [Project management workspace in Notion](https://www.notion.so/Homepage-2ff744c443814f459d80a6e5819226a5)

* [Loom video about the project](https://www.loom.com/share/365d412db3a0474ba46d4fdd7f4c5494)

* [Medium article about this project](https://towardsdatascience.com/detecting-deforestation-from-satellite-images-7aa6dfbd9f61)

This is the result of a group project, made by [André Ferreira](https://andrecnf.com/) and [Karthik Bhaskar](https://www.kbhaskar.com/), for the [Full Stack Deep Learning - Spring 2021 online course](http://fullstackdeeplearning.com/spring2021/).

## Very first steps

### Initial

1. Initialize `git` inside your repo:

```bash
git init
```

2. If you don't have `Poetry` installed run:

```bash
make download-poetry
```

3. Initialize poetry and install `pre-commit` hooks:

```bash
make install
```

4. Upload initial code to GitHub (ensure you've run `make install` to use `pre-commit`):

```bash
git add .
git commit -m ":tada: Initial commit"
git branch -M main
git remote add origin https://github.com/karthikraja95/fsdl_deforestation_detection.git
git push -u origin main
```

### Initial setting up

- Set up [Dependabot](https://docs.github.com/en/github/administering-a-repository/enabling-and-disabling-version-updates#enabling-github-dependabot-version-updates) to ensure you have the latest dependencies.
- Set up [Stale bot](https://github.com/apps/stale) for automatic issue closing.

### Poetry

All manipulations with dependencies are executed through Poetry. If you're new to it, look through [the documentation](https://python-poetry.org/docs/).

<details>
<summary>Notes about Poetry</summary>
<p>

Poetry's [commands](https://python-poetry.org/docs/cli/#commands) are very intuitive and easy to learn, like:

- `poetry add numpy`
- `poetry run pytest`
- `poetry build`
- etc

</p>
</details>

### Makefile usage

[`Makefile`](https://github.com/karthikraja95/fsdl_deforestation_detection/blob/master/Makefile) contains many functions for fast assembling and convenient work.

<details>
<summary>1. Download Poetry</summary>
<p>

```bash
make download-poetry
```

</p>
</details>

<details>
<summary>2. Install all dependencies and pre-commit hooks</summary>
<p>

```bash
make install
```

If you do not want to install pre-commit hooks, run the command with the NO_PRE_COMMIT flag:

```bash
make install NO_PRE_COMMIT=1
```

</p>
</details>

<details>
<summary>3. Check the security of your code</summary>
<p>

```bash
make check-safety
```

This command launches a `Poetry` and `Pip` integrity check as well as identifies security issues with `Safety` and `Bandit`. By default, the build will not crash if any of the items fail. But you can set `STRICT=1` for the entire build, or you can configure strictness for each item separately.

```bash
make check-safety STRICT=1
```

or only for `safety`:

```bash
make check-safety SAFETY_STRICT=1
```

multiple

```bash
make check-safety PIP_STRICT=1 SAFETY_STRICT=1
```

> List of flags for `check-safety` (can be set to `1` or `0`): `STRICT`, `POETRY_STRICT`, `PIP_STRICT`, `SAFETY_STRICT`, `BANDIT_STRICT`.

</p>
</details>

<details>
<summary>4. Check the codestyle</summary>
<p>

The command is similar to `check-safety` but to check the code style, obviously. It uses `Black`, `Darglint`, `Isort`, and `Mypy` inside.

```bash
make check-style
```

It may also contain the `STRICT` flag.

```bash
make check-style STRICT=1
```

> List of flags for `check-style` (can be set to `1` or `0`): `STRICT`, `BLACK_STRICT`, `DARGLINT_STRICT`, `ISORT_STRICT`, `MYPY_STRICT`.

</p>
</details>

<details>
<summary>5. Run all the codestyle formaters</summary>
<p>

Codestyle uses `pre-commit` hooks, so ensure you've run `make install` before.

```bash
make codestyle
```

</p>
</details>

<details>
<summary>6. Run tests</summary>
<p>

```bash
make test
```

</p>
</details>

<details>
<summary>7. Run all the linters</summary>
<p>

```bash
make lint
```

the same as:

```bash
make test && make check-safety && make check-style
```

> List of flags for `lint` (can be set to `1` or `0`): `STRICT`, `POETRY_STRICT`, `PIP_STRICT`, `SAFETY_STRICT`, `BANDIT_STRICT`, `BLACK_STRICT`, `DARGLINT_STRICT`, `ISORT_STRICT`, `MYPY_STRICT`.

</p>
</details>

<details>
<summary>8. Build docker</summary>
<p>

```bash
make docker
```

which is equivalent to:

```bash
make docker VERSION=latest
```

More information [here](https://github.com/karthikraja95/fsdl_deforestation_detection/tree/master/docker).

</p>
</details>

<details>
<summary>9. Cleanup docker</summary>
<p>

```bash
make clean_docker
```

or to remove all build

```bash
make clean
```

More information [here](https://github.com/karthikraja95/fsdl_deforestation_detection/tree/master/docker).

</p>
</details>

## 🛡 License

[![License](https://img.shields.io/github/license/karthikraja95/fsdl_deforestation_detection)](https://github.com/karthikraja95/fsdl_deforestation_detection/blob/master/LICENSE)

This project is licensed under the terms of the `MIT` license. See [LICENSE](https://github.com/karthikraja95/fsdl_deforestation_detection/blob/master/LICENSE) for more details.

## 📃 Citation

```
@misc{fsdl_deforestation_detection,
  author = {Karthik Bhaskar, Andre Ferreira},
  title = {Predicting deforestation from Satellite Images},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/karthikraja95/fsdl_deforestation_detection}}
}
```


