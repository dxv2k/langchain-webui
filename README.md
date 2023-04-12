# LangChain-Gradio Template


## To run:
```
python app.py
```

Reload: 
```
gradio appp.py 
```

This repo serves as a template for how to deploy a LangChain on Gradio.
This is particularly useful because you can easily deploy Gradio apps on Hugging Face spaces, 
making it very easy to share you LangChain applications on there.

This repo contains an `app.py` file which has a template for a chatbot implementation.
This was heavily inspired [James Weaver's examples](https://huggingface.co/JavaFXpert).

## Adding your chain
To add your chain, you need to change the `load_chain` function in `app.py`.
Depending on the type of your chain, you may also need to change the inputs/outputs that occur later on (in the `chat` function).

## Deploying to Hugging Face spaces
To deploy to Hugging Face spaces, you should first create a new Gradio space on Hugging Face.
From there, you can then copy over the `app.py` file here, and the `requirements.txt`.
Do not copy over this ReadMe - Hugging Face spaces requires a ReadMe in a particular format.



# Commit Convention: 
* API relevant changes
    * `feat` Commits, that adds a new feature
    * `fix` Commits, that fixes a bug
* `refactor` Commits, that rewrite/restructure your code, however does not change any behaviour
    * `perf` Commits are special `refactor` commits, that improve performance
* `style` Commits, that do not affect the meaning (white-space, formatting, missing semi-colons, etc)
* `test` Commits, that add missing tests or correcting existing tests
* `docs` Commits, that affect documentation only
* `build` Commits, that affect build components like build tool, ci pipeline, dependencies, project version, ...
* `ops` Commits, that affect operational components like infrastructure, deployment, backup, recovery, ...
* `chore` Miscellaneous commits e.g. modifying `.gitignore`
